# Data dictionary

| Column | Meaning |
|---|---|
| DateTimeZA | Transaction timestamp in local South African time. |
| TransactionId | Basket identifier; multiple rows may share it. |
| LineItemSequence | Line sequence within a basket. |
| LocationId | Source store/site identifier. |
| ID | Source anonymized customer identifier; read as a string. |
| LineItemType | Source line type; fuel regression uses Sale_Fuel. |
| TransactionTotalInclAmount | Whole basket total repeated on its lines. Do not sum this column over lines. |
| StockItemId | Source product identifier. |
| ItemName | Source product or adjustment name. |
| ItemCategoryLevel1 | Source coarse category. |
| ItemCategoryLevel2 | Category used for target rules; literal NA is not missing. |
| ItemCategoryLevel3 | Source finer category. |
| Quantity | Signed quantity; litres for Sale_Fuel, source units otherwise. |
| TotalExclAmount | Source signed line amount; store pre-tax, source fuel includes tax. Used directly under label rules. |
| TotalModifierAmount | Source price modifier; do not subtract again when constructing targets. |
| UnitSellingExclPrice | Source unit price; do not reconstruct target amount by multiplying and subtracting modifiers. |
