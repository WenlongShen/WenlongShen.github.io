 $(document).ready(function () {
        var time1 = 0;
        var show = false;
        var names = new Array(); //文章名字等
        var urls = new Array(); //文章地址

        $("#cb-close-btn").click(function () {
            $(".cb-search-tool").css("display", "none");
            show = false;
            time1 = 0;
        });

        $("#cb-search-btn").click(function(){
            $(".cb-search-tool").css("display", "block");
            show = true;
            $("#cb-search-content").val("");
            $("#cb-search-content").focus();
            time1 = 0;
        });

        $.getJSON("/search/cb-search.json").done(function (data) {
            if (data.code == 0) {
                for (var index in data.data) {
                    var item = data.data[index];
                    names.push(item.title + "-" + item.content);
                    urls.push(item.url);
                }

                $("#cb-search-content").typeahead({
                    source: names,

                    afterSelect: function (item) {
						$(".cb-search-tool").css("display", "none");
                        show = false;
                        window.location.href = (urls[names.indexOf(item)]);
                        return item;
                    }
                });
            }
        }).error(function(data, b) { console.log("json解析错误，搜索功能暂不可用"); });

    });
