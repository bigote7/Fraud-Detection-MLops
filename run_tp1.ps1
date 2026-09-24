python donnees.py
foreach ($p in '100 4', '200 8', '300 12', '500 16', '600 24') {
  $a = $p -split ' '
  python src/train.py $a[0] $a[1]
}
Write-Host 'TP1 termine - consultez http://127.0.0.1:5000'
