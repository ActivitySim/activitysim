
# rst doesn't support image maps, so we'll add one after the html has been built

fileName = "_build/html/abmexample.html"
line = 'src="_images/abmexample.jpg" />'
lineWithMap = 'src="_images/abmexample.jpg" usemap="#image-map" />'
imageMap = """\n   <map name="image-map">
      <area target="" alt="" title="" href="models.html#accessibility" coords="235,3,443,30" shape="rect">
      <area target="" alt="" title="" href="models.html#school-location" coords="445,72,234,44" shape="rect">
      <area target="" alt="" title="" href="models.html#auto-ownership" coords="238,84,445,112" shape="rect">
      <area target="" alt="" title="" href="models.html#free-parking-eligibility" coords="239,124,442,151" shape="rect">
      <area target="" alt="" title="" href="models.html#cdap" coords="236,163,442,192" shape="rect">
      <area target="" alt="" title="" href="models.html#mandatory-tour-frequency" coords="163,269,7,241" shape="rect">
      <area target="" alt="" title="" href="models.html#mandatory-tour-scheduling" coords="5,400,161,431" shape="rect">
      <area target="" alt="" title="" href="models.html#tour-mode" coords="5,471,161,506" shape="rect">
      <area target="" alt="" title="" href="models.html#joint-tour-frequency" coords="177,241,334,273" shape="rect">
      <area target="" alt="" title="" href="models.html#joint-tour-composition" coords="178,284,327,311" shape="rect">
      <area target="" alt="" title="" href="models.html#joint-tour-participation" coords="178,323,331,350" shape="rect">
      <area target="" alt="" title="" href="models.html#joint-tour-destination-choice" coords="178,361,329,389" shape="rect">
      <area target="" alt="" title="" href="models.html#joint-tour-scheduling" coords="179,403,332,430" shape="rect">
      <area target="" alt="" title="" href="models.html#joint-tour-mode-choice" coords="173,443,331,465" shape="rect">
      <area target="" alt="" title="" href="models.html#non-mandatory-tour-frequency" coords="346,245,496,268" shape="rect">
      <area target="" alt="" title="" href="models.html#non-mandatory-tour-destination-choice" coords="344,361,496,389" shape="rect">
      <area target="" alt="" title="" href="models.html#non-mandatory-tour-scheduling" coords="346,401,499,426" shape="rect">
      <area target="" alt="" title="" href="models.html#man-non-man-tour-mode-choice" coords="344,476,501,500" shape="rect">
      <area target="" alt="" title="" href="models.html#atwork-subtour-frequency" coords="518,243,668,269" shape="rect">
      <area target="" alt="" title="" href="models.html#atwork-subtour-destination" coords="513,364,668,387" shape="rect">
      <area target="" alt="" title="" href="models.html#atwork-subtour-scheduling" coords="517,402,667,428" shape="rect">
      <area target="" alt="" title="" href="models.html#atwork-subtour-mode-choice" coords="516,443,669,468" shape="rect">
      <area target="" alt="" title="" href="models.html#intermediate-stop-frequency" coords="669,549,8,522" shape="rect">
      <area target="" alt="" title="" href="models.html#trip-purpose" coords="233,575,438,598" shape="rect">
      <area target="" alt="" title="" href="models.html#trip-destination-choice" coords="236,617,436,641" shape="rect">
      <area target="" alt="" title="" href="models.html#trip-scheduling" coords="236,657,437,680" shape="rect">
      <area target="" alt="" title="" href="models.html#trip-mode-choice" coords="238,696,436,718" shape="rect">
      <area target="" alt="" title="" href="models.html#trip-cbd-parking" coords="236,730,434,757" shape="rect">
    </map>
    """  # noqa

print("add image map to " + fileName)

with open(fileName) as file:
    lines = file.readlines()

with open(fileName, 'w') as file:
    for l in lines:
        if line in l:
            print("updated " + fileName)
            file.writelines("%s" % l.replace(line, lineWithMap))
            file.writelines("%s" % imageMap)
        else:
            file.writelines("%s" % l)
