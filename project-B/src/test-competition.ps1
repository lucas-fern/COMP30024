$n_games = 100
$filepath = "competition/HA1-5-1-1_HA1-5-5-1.txt"

Remove-Item -path $filepath

for ($i=1; $i -le $n_games; $i++)
{
    Write-Host $i
    $OUTPUT=$(python -m referee heuristic_agent heuristic_agent_2 | tail -n 1)
    Add-Content $filepath $OUTPUT
}
