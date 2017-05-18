function getRTime(){
	var StartTime= new Date('2016/02/22 08:00:00');
	var NowTime = new Date();
	var t = NowTime.getTime() - StartTime.getTime();

	var y = NowTime.getFullYear() - StartTime.getFullYear() + 1;
	var d = Math.floor(t/1000/60/60/24);

	document.getElementById("t_y").innerHTML = "<font size=3>"+ y + "年</font>";
	document.getElementById("t_d").innerHTML = "<font size=3>"+ d + "天</font>";
}
setInterval(getRTime,1000);