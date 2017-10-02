rm tmpresults
rm f1results
for i in $( ls Models ); do
    th eval.lua -model Models/$i
    python quantify_results.py --file tmpresults --label $i --confusionMatrix $i;
done
