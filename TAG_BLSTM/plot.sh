set yrange [56:70]
set grid
plot "tmpresults" title " TAG_BLSTM: Accuracy vs #epochs\n hiddensize = 50 (bidirectional for sent)\n hiddensize2 = 5 (unidirectional for tags)\n format: seprate files for words and tags (reduced)\n batchsize = 7" smooth csplines
