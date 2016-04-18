import pandas as pd
import string

f = pd.read_excel(open("../ModeChoice.xls"), sheetname=None)

coeffs = {}
coeff_ind = None
specs = {}
ind = None

for key, df in f.iteritems():
    if "debug" in key or "model" in key or "data" in key:
        continue
    print key

    """
    Really these lines just extract the relevant parts out of the spreadsheet
    """

    # the headers are actually split up among a couple of rows (ouch)
    df.columns = list(df.iloc[1].values)[:6] + list(df.iloc[2].values)[6:]

    filt = '26 < = No <= 57' if key != "WorkBased" else '1 < = No <= 36'
    coeffs[key] = df.query(filt).set_index('Token')['Formula for variable']
    if coeff_ind is None:
        coeff_ind = coeffs[key].index

    df = df.iloc[2:]

    # this is not the best - there are a couple of cases where the specs
    # differ across segments and we want them to be the same FOR NOW - they
    # can differ once we get them all lined up
    if key == "Escort":
        df = df.query("No != 401 and No >= 123")
    elif key == "WorkBased":
        df = df.query("No != 407 and No >= 126")
    else:
        df = df.query("No >= 123")

    df = df.drop(['No', 'Token', 'Filter', 'Index'], axis=1)
    df.columns = ['Description', 'Expression'] + list(df.columns[2:])
    df.set_index(['Description', 'Expression'], inplace=True)

    # these lines merge the alternatives that are used
    # into a comma separated list
    alt_l = []
    val_l = []
    for _, row in df.iterrows():
        alts = list(row.dropna().index)
        vals = list(row.dropna())

        # assert only 1 unique value
        if len(vals) == 0:
            vals = [0]
        assert len(set(vals)) == 1
        val = vals[0]

        alt_l.append(string.join(alts, ","))
        val_l.append(val)

    #    print alts
    df = pd.DataFrame({
        'Alternatives': alt_l,
        key: val_l
    }, index=df.index).set_index('Alternatives', append=True)

    if ind is None:
        ind = df.index

    assert len(ind) == len(df)

    # ok, this is a bit bizarre, but it appears to me the coefficients are in
    # the expressions column so you can just write 1s in the cells - if we're
    # going to stack the columns we need to move the coeffs back to the cells
    df = df.reset_index()

    # tmp = df.Expression
    # df["Expression"].iloc[232:] = df.iloc[232:][key]
    df[key].iloc[232:] = df["Expression"].iloc[232:]

    specs[key] = df[key].values

df = pd.DataFrame(specs)
df.index = ind

df.to_csv('tour_mode_choice.csv')

pd.DataFrame(coeffs).loc[coeff_ind].to_csv('tour_mode_choice_coeffs.csv')
