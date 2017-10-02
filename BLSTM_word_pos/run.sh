rm tmpresults
for i in $( ls Models ); do
    th eval.lua -model Models/$i -testfile cvTest
    python quantify_results.py --file tmpresults --label  Models/$i;
done

sed -i 's/.*el//g' tmpresults
sed -i 's/_.* / /g' tmpresults

