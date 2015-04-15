import pandas as pd

f = pd.read_excel(open("../ModeChoice.xls"), sheetname=None)

dfs = {}
alts = None

for key, df in f.iteritems():
    if "debug" in key or "model" in key or "data" in key:
        continue

    # this is not the best - there are a couple of cases where the specs
    # differ across segments and we want them to be the same FOR NOW - they
    # can differ once we get them all lined up
    if key == "Escort":
        df = df.query("Model != 401")
    if key == "WorkBased":
        df = df.query("Model != 407")

    # the headers are actually split up among a couple of rows (ouch)
    df.columns = list(df.iloc[1].values)[:6] + list(df.iloc[2].values)[6:]

    df = df.iloc[131:]
    df = df.drop(['No', 'Token', 'Filter', 'Index'], axis=1)
    df.columns = ['Description', 'Expression'] + list(df.columns[2:])
    df.set_index(['Description', 'Expression'], inplace=True)
    df = df.stack()

    # ok, this is a bit bizarre, but it appears to me the coefficients

    ind = df.index

    cur_alts = list(df.reset_index()["level_2"].values)
    print len(df)
    if alts:
        if len(df) != len(alts):
            print "ERROR, segment %s has different number of alternatives" % key
            continue
        # error checking - make sure series alternatives are the same
        assert alts == cur_alts
    alts = cur_alts

    dfs[key] = df.reset_index(drop=True)

df = pd.DataFrame(dfs)
df.index = ind
df.index.names = df.index.names[:2] + ["Alternative"]

df.to_csv('tour_mode_choice.csv')