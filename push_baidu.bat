
for /l %%i in (1,1,50) do curl -H 'Content-Type:text/plain' --data-binary @urls.txt "http://data.zz.baidu.com/urls?site=https://wenlongshen.github.io&token=1nYEnh4jhunI9erR"

pause