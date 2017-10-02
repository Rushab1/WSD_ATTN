set yrange [65:82]
set term pdf
set grid

plot "tmpresults" title "VALIDATION:\n\nBLSTM: Accuracy vs #epochs\nhiddensize = 50 (birectional)\nformat: words (no tags)\nbatchsize = 7" smooth csplines ,\
    "tmptest" title "TEST:\n" smooth csplines
