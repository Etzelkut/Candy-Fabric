import androidhelper
while True:
	with open('/storage/emulated/0/proard/command.txt', 'r') as file:   	
		data = file.readlines()
	if len(data)>1:
		if data[0] == "1\n":
				droid = androidhelper.Android()
				droid.cameraCapturePicture('/storage/emulated/0/DCIM/pro/pro.jpg')
				f = open("/storage/emulated/0/proard/command.txt",'w')
				f.write('0\n')
				f.write('1\n')
				f.close()
		

