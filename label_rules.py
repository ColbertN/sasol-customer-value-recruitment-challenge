"""Reproducible Sasol customer labels. Reads only the CSV supplied by the caller."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import numpy as np
import pandas as pd

ID = 'ID'
SCALE = 1000000

def load_transactions(path):
    return prepare(pd.read_csv(path, dtype=str, keep_default_na=False))

def prepare(raw):
    d = raw.copy()
    d['_time'] = pd.to_datetime(d.DateTimeZA, format='mixed', errors='raise')
    for name, key in [('Quantity', '_quantity'), ('TotalExclAmount', '_amount')]:
        values = {}
        for v in d[name].unique():
            scaled = Decimal(v) * SCALE
            if scaled != scaled.to_integral_value():
                raise ValueError(f'{name} has precision beyond six decimal places')
            values[v] = int(scaled)
        d[key] = d[name].map(values).astype('int64')
    return d

def eligible_ids(d, cutoff, minimum):
    n = d[d._time < pd.Timestamp(cutoff)].groupby(ID).TransactionId.nunique()
    return n[n >= minimum].index.sort_values()

def raw_labels(d, cutoff, config):
    t = pd.Timestamp(cutoff)
    end = t + pd.DateOffset(months=3)
    ids = eligible_ids(d, t, config['minimum_historical_transactions'])
    p = d[~d.ItemName.isin(config['adjustment_items'])]
    h = p[p._time < t]
    recent = h[h._time >= t - pd.DateOffset(months=3)]
    future = p[(p._time >= t) & (p._time < end)]
    mapping = config['category_mapping']
    cats = sorted(c for c, v in mapping.items() if v['included'])
    unknown = set(p.ItemCategoryLevel2) - set(mapping)
    if unknown:
        raise ValueError(f'Unmapped source categories: {sorted(unknown)}')
    def matrix(frame, count=False):
        frame = frame[frame.ItemCategoryLevel2.isin(cats)]
        if count:
            frame = frame[(frame._quantity > 0) & (frame._amount > 0)]
        g = frame.groupby([ID, 'ItemCategoryLevel2'])
        s = g.TransactionId.nunique() if count else g._amount.sum()
        return s.unstack(fill_value=0).reindex(index=ids, columns=cats, fill_value=0).astype('int64')
    historical = matrix(h, True)
    future_purchases = matrix(future, True)
    previous_spend = matrix(recent)
    future_spend = matrix(future)
    delta = future_spend - previous_spend
    grow = historical.gt(0) & future_purchases.ge(config['growth_min_purchases'])
    grow &= delta.ge(int(Decimal(str(config['growth_min_rand'])) * SCALE))
    if config['growth_min_relative'] is not None:
        grow &= future_spend.gt(previous_spend.clip(lower=0) * (1 + config['growth_min_relative']))
    adopt = historical.eq(0) & future_purchases.ge(config['adoption_min_purchases']) & future_spend.gt(0)
    threshold = int(Decimal(str(config['adoption_min_rand'])) * SCALE)
    adopt &= future_spend.ge(threshold)
    av = future_spend.where(adopt, 0).to_numpy()
    gv = delta.where(grow, 0).to_numpy()
    am, gm = av.max(axis=1), gv.max(axis=1)
    is_adopt = (am > 0) & (am >= gm)
    is_growth = (gm > 0) & ~is_adopt
    labels = np.full(len(ids), 'Stable', dtype=object)
    winner = np.full(len(ids), '', dtype=object)
    increment = np.zeros(len(ids), dtype='int64')
    for mask, values, kind in [(is_adopt, av, 'New category adoption'), (is_growth, gv, 'Existing-category growth')]:
        cat = np.array(cats)[values.argmax(axis=1)]
        winner[mask] = cat[mask]
        increment[mask] = values.max(axis=1)[mask]
        labels[mask] = [kind + ': ' + mapping[c][kind] for c in cat[mask]]
    labels[future_purchases.sum(axis=1).eq(0)] = 'Inactivity'
    fuel = future[future.ItemCategoryLevel2.eq('Fuel') & future.LineItemType.eq('Sale_Fuel')].groupby(ID)._quantity.sum().reindex(ids, fill_value=0)
    nonfuel = future_spend.drop(columns=['Fuel']).sum(axis=1)
    result = pd.DataFrame({ID: ids, 'fuel_litres': fuel.to_numpy()/SCALE,
                           'nonfuel_rand': nonfuel.to_numpy()/SCALE, 'Opportunity': labels,
                           'winning_source_category': winner, 'winning_increment_rand': increment/SCALE})
    if (result[['fuel_litres', 'nonfuel_rand']] < 0).any().any():
        raise ValueError('Negative net customer target: requires an explicit policy; not silently clipped')
    if not set(labels) <= set(config['opportunity_labels']):
        raise ValueError('Unexpected opportunity label')
    return result

def transform(raw, config):
    out = raw[[ID]].copy()
    for target, source in [('CLV_fuel', 'fuel_litres'), ('CLV_nonfuel', 'nonfuel_rand')]:
        out[target] = np.log1p(raw[source].to_numpy()) / config['normalization'][target]
    out['Opportunity'] = raw.Opportunity.to_numpy()
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--transactions', required=True)
    ap.add_argument('--cutoff', required=True)
    ap.add_argument('--config', default=str(Path(__file__).with_name('label_config.json')))
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text())
    end = pd.Timestamp(args.cutoff) + pd.DateOffset(months=3)
    if end > pd.Timestamp(config['public_history_end_exclusive']):
        raise ValueError('The requested full outcome window is outside the released history')
    result = transform(raw_labels(load_transactions(args.transactions), args.cutoff, config), config)
    result.to_csv(args.output, index=False, float_format='%.17g')
    print(f'Created {len(result)} customer labels')

if __name__ == '__main__':
    main()
