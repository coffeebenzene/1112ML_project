# 01.112 Machine Learning Project
NLP with Hidden Markov Model

Group:
* Eric Teo Zhi Han 1001526
* Jo Hsi Keong 1001685

## How to run scripts
There is a script for each part (2,3,4,5) of the project.

The syntax for each script is as follows:
```
python part#.py -t trainfile -i infile -o outfile -f folder [-d]
```

Where:
 * `#` in `part#.py` is the part number
 * `trainfile` is the name of the file used for training (defaults to `train`)
 * `infile` is the name of the file with validation data to be annotated (defaults to `dev.in`)
 * `outfile` is the name of the output file to write annotated data to (defaults to `dev.p#.out`)
 * `folder` is the path of the folder where `trainfile`, `infile` and `outfile` are at. (defaults to `.` the current folder)  
   This makes it more convenient to run the script. No need write the full path of files. (i.e. `trainfile`, `infile` and `outfile` are relative paths from `folder`.)
 * `-d` is an option for debug mode. The script will print (large amounts of) debug info while calculating.

Example execution:
```
python part4.py -f "C:\Users\Eric\ML_project\EN" -d
```
Assuming that the EN files are stored in the `C:\Users\Eric\ML_project\EN` folder, This will run the part4 script on the EN files with debug information. The output will be in `C:\Users\Eric\ML_project\EN\dev.p4.out`.

Notes: The script was tested in windows. It should work under linux, but we did not check.

## F-score Results

<table>
<tr>
    <th></th>
    <th colspan=2>EN</th>
    <th colspan=2>FR</th>
    <th colspan=2>SG</th>
    <th colspan=2>CN</th>
</tr>
<tr>
    <th>Part</th>
    <th>Entity F-score</th><th>Sentiment F-score</th>
    <th>Entity F-score</th><th>Sentiment F-score</th>
    <th>Entity F-score</th><th>Sentiment F-score</th>
    <th>Entity F-score</th><th>Sentiment F-score</th>
</tr>
<tr>
    <td>Part&nbsp2</td>
    <td>0.2313</td><td>0.0995</td>
    <td>0.2653</td><td>0.0991</td>
    <td>0.1990</td><td>0.0789</td>
    <td>0.0995</td><td>0.0310</td>
</tr>
<tr>
    <td>Part&nbsp3</td>
    <td>0.5361</td><td>0.3299</td>
    <td>0.5758</td><td>0.3702</td>
    <td>0.3667</td><td>0.2318</td>
    <td>0.2462</td><td>0.1808</td>
</tr>
<tr>
    <td>Part&nbsp4</td>
    <td>0.5387</td><td>0.3441</td>
    <td>0.5707</td><td>0.3687</td>
    <td>0.3639</td><td>0.2383</td>
    <td>0.2432</td><td>0.1706</td>
</tr>
</table>
