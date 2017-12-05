resultfile="raw eval result.txt"
echo -n "" > "$resultfile";

for i in 2 3 4 5;
do
    echo -e "====Part $i --------------------\r" >> "$resultfile"; # \r for windows!
    for lang in "EN" "FR" "SG" "CN";
    do
        echo -e "\r\n----$lang :\r" >> "$resultfile";
        py EvalScript/evalResult.py $lang/dev.out $lang/dev.p$i.out >> "$resultfile";
    done;
done;