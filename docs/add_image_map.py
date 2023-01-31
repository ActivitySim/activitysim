# rst doesn't support image maps, so we'll add one after the html has been built

fileName = "_build/html/examples.html"
line = 'src="_images/abmexample.jpg" />'
lineWithMap = 'src="_images/abmexample.jpg" usemap="#image-map" />'
imageMap = """\n   <map name="image-map">
    <area target="" alt="" title="" href="models.html#write-trip-matrices" coords="234,683,436,709" shape="rect">
    <area target="" alt="" title="" href="models.html#trip-mode-choice" coords="438,670,235,646" shape="rect">
    <area target="" alt="" title="" href="models.html#trip-scheduling" coords="235,607,434,630" shape="rect">
    <area target="" alt="" title="" href="models.html#trip-destination-choice" coords="235,565,433,588" shape="rect">
    <area target="" alt="" title="" href="models.html#trip-purpose" coords="236,524,436,550" shape="rect">
    <area target="" alt="" title="" href="models.html#intermediate-stop-frequency" coords="5,481,663,505" shape="rect">
    <area target="" alt="" title="" href="models.html#atwork-subtour-scheduling" coords="516,401,666,427" shape="rect">
    <area target="" alt="" title="" href="models.html#atwork-subtour-destination" coords="517,361,667,385" shape="rect">
    <area target="" alt="" title="" href="models.html#atwork-subtour-frequency" coords="515,243,667,265" shape="rect">
    <area target="" alt="" title="" href="models.html#non-mandatory-tour-scheduling" coords="347,404,497,425" shape="rect">
    <area target="" alt="" title="" href="models.html#non-mandatory-tour-destination-choice" coords="347,363,496,384" shape="rect">
    <area target="" alt="" title="" href="models.html#non-mandatory-tour-frequency" coords="347,245,495,267" shape="rect">
    <area target="" alt="" title="" href="models.html#joint-tour-scheduling" coords="175,402,327,424" shape="rect">
    <area target="" alt="" title="" href="models.html#joint-tour-destination-choice" coords="175,361,326,386" shape="rect">
    <area target="" alt="" title="" href="models.html#joint-tour-participation" coords="175,324,326,345" shape="rect">
    <area target="" alt="" title="" href="models.html#joint-tour-composition" coords="176,282,326,307" shape="rect">
    <area target="" alt="" title="" href="models.html#joint-tour-frequency" coords="176,243,327,266" shape="rect">
    <area target="" alt="" title="" href="models.html#tour-mode" coords="5,442,667,464" shape="rect">
    <area target="" alt="" title="" href="models.html#mandatory-tour-scheduling" coords="5,401,156,425" shape="rect">
    <area target="" alt="" title="" href="models.html#mandatory-tour-frequency" coords="5,241,157,267" shape="rect">
    <area target="" alt="" title="" href="models.html#cdap" coords="235,165,434,189" shape="rect">
    <area target="" alt="" title="" href="models.html#free-parking-eligibility" coords="239,126,435,148" shape="rect">
    <area target="" alt="" title="" href="models.html#auto-ownership" coords="237,86,433,109" shape="rect">
    <area target="" alt="" title="" href="models.html#school-location" coords="237,43,435,67" shape="rect">
    <area target="" alt="" title="" href="models.html#accessibility" coords="236,6,436,27" shape="rect">
    <area target="" alt="" title="" href="models.html#work-from-home" coords="500,25,675,53" shape="rect">
    <area target="" alt="" title="" href="models.html#transit-pass-ownership" coords="501,67,676,95" shape="rect">
    <area target="" alt="" title="" href="models.html#telecommute-frequency" coords="502,108,677,136" shape="rect">
   </map>
    """  # noqa

print("add image map to " + fileName)

with open(fileName, encoding="utf-8") as file:
    lines = file.readlines()

with open(fileName, "w") as file:
    for aLine in lines:
        if line in aLine:
            print("updated " + fileName)
            file.writelines("%s" % aLine.replace(line, lineWithMap))
            file.writelines("%s" % imageMap)
        else:
            file.writelines("%s" % aLine)
