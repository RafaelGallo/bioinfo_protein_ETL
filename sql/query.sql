-- accuracy by split / acerto por conjunto
SELECT split, COUNT(*) AS proteins, ROUND(AVG(correct), 3) AS accuracy
FROM predictions_xgb
GROUP BY split;

-- test proteins predicted nuclear with protein names / proteínas do teste previstas como nucleares com nomes
SELECT p.entry, u.protein_names, u.organism, p.proba
FROM predictions_xgb p
JOIN uniprot_target u ON u.entry = p.entry
WHERE p.split = 'test' AND p.pred = 1
ORDER BY p.proba DESC
LIMIT 20;