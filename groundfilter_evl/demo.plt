set terminal pdf font "Times New Roman,16" #transparent enhanced font "arial,10" fontscale 1.0 size 600, 400 
set grid
set size 1,0.8
set output 'compare.pdf'
set style fill solid 0.5 border -1
set style boxplot outliers pointtype 6
set style data boxplot
set boxwidth  0.5
set pointsize 0.5
unset key
set ylabel 'Remained Point Ratio'
set border 2
#set xtics ("KITTI_{FULL}" 1,"KITTI_{FOV}" 2,"RS128_{FULL}" 3,"RS128_{FOV}" 4,) scale 0.0
set xtics ("KITTI_{NG}" 1,"RS_{NG}" 2,"KITTI_{FOV}" 3,"RS_{FOV}" 4,"KITTI_{FOV, NG}" 5,"RS_{FOV, NG}" 6) scale 0.0 font "Times New Roman,12"
set xtics nomirror
set ytics nomirror
set yrange [0:0.8]
plot 'kitti.txt' using (1):1,\
	 'rs-ruby.txt' using (2):1,\
     'kitti.txt' using (3):2,\
	 'rs-ruby.txt' using (4):2,\
	 'kitti.txt' using (5):3,\
     'rs-ruby.txt' using (6):3
	 