function getRTime(){
	var NowTime = new Date();
	// 相识
	var StartTimeK = new Date('2016/02/22 08:00:00');
	var k = NowTime.getTime() - StartTimeK.getTime();
	var yk = NowTime.getFullYear() - StartTimeK.getFullYear() + 1;
	var dk = Math.floor(k/1000/60/60/24);
	// 领证
	var StartTimeL = new Date('2017/11/27 14:00:00');
	var l = NowTime.getTime() - StartTimeL.getTime();
	var yl = NowTime.getFullYear() - StartTimeL.getFullYear() + 1;
	var dl = Math.floor(l/1000/60/60/24);
	// 婚礼
	var StartTimeW = new Date('2018/03/16 12:00:00');
	var w = NowTime.getTime() - StartTimeW.getTime();
	var yw = NowTime.getFullYear() - StartTimeW.getFullYear() + 1;
	var dw = Math.floor(w/1000/60/60/24);

	document.getElementById("yk").innerHTML = "<font size=3>"+ yk + "年</font>";
	document.getElementById("dk").innerHTML = "<font size=3>"+ dk + "天</font>";
	document.getElementById("yl").innerHTML = "<font size=3>"+ yl + "年</font>";
	document.getElementById("dl").innerHTML = "<font size=3>"+ dl + "天</font>";
	document.getElementById("yw").innerHTML = "<font size=3>"+ yw + "年</font>";
	document.getElementById("dw").innerHTML = "<font size=3>"+ dw + "天</font>";
}
setInterval(getRTime,1000);