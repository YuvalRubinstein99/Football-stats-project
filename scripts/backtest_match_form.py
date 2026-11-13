"""Replay unchanged betting strategies on verified chronological form predictions."""
import csv
import gzip
import hashlib
import html
import json
from pathlib import Path
from select_expanded_features import ROOT, pd, np
from live_bets import integrated_recommendations, STRATEGIES, SEARCH_CANDIDATE
from stake_levels import size_bet

DATA = ROOT / 'results/match_form'
OUT = DATA / 'betting'
RULES = STRATEGIES + [SEARCH_CANDIDATE]
MODES = ['Flat $1 original rule', 'Flat $1 positive edge', 'Existing $1-$5 tiers']


def settle_arrays(frame, probabilities):
    """One unit split across outcomes, with the original A/D/H tie conventions."""
    p = np.asarray(probabilities, float)
    odds = frame[['B365H', 'B365D', 'B365A']].to_numpy(float)
    if p.shape != odds.shape or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
        raise ValueError('Invalid or misaligned H/D/A probabilities')
    if not np.isfinite(odds).all() or (odds <= 1).any():
        raise ValueError('Invalid odds')
    outcomes = (1 - np.sign(frame.home_score - frame.away_score)).to_numpy(int)
    shape = (len(frame), len(RULES))
    result = {k: np.zeros(shape) for k in
              ['original_stake', 'eligible_stake', 'tier_stake', 'unit_return',
               'home_weight', 'draw_weight', 'away_weight', 'unavailable']}
    for i, row in enumerate(frame.itertuples()):
        match = dict(home=row.home_team_name, away=row.away_team_name,
                     p_home=p[i,0], p_draw=p[i,1], p_away=p[i,2],
                     B365H=odds[i,0], B365D=odds[i,1], B365A=odds[i,2])
        bets = integrated_recommendations(match)
        if [b['strategy'] for b in bets] != RULES:
            raise ValueError('Strategy order changed')
        for j, bet in enumerate(bets):
            weights = [bet[s + '_stake'] for s in ('home','draw','away')]
            amount = size_bet(bet)['stake_dollars']
            original = bet['action'] == 'Bet'
            if not np.isclose(sum(weights), int(original)):
                raise ValueError('Strategy stake weights do not sum to one')
            result['original_stake'][i,j] = int(original)
            result['eligible_stake'][i,j] = int(amount > 0)
            result['tier_stake'][i,j] = amount
            result['unit_return'][i,j] = weights[outcomes[i]] * odds[i,outcomes[i]]
            for k, side in enumerate(('home','draw','away')):
                result[side + '_weight'][i,j] = weights[k]
            result['unavailable'][i,j] = bet['action'] == 'Unavailable'
    return result


def summarize(dates, stake, unit_return):
    stake, unit_return = np.asarray(stake), np.asarray(unit_return)
    gross = stake * unit_return
    net = gross - stake
    daily = pd.DataFrame(dict(date=np.asarray(dates), stake=stake, gross=gross, net=net))
    daily = daily.groupby('date', sort=True)[['stake','gross','net']].sum()
    cumulative = daily.net.cumsum().to_numpy()
    peak = np.maximum.accumulate(np.r_[0., cumulative])[1:]
    summary = dict(matches=len(stake), bets=int((stake > 0).sum()),
        wins=int(((stake > 0) & (unit_return > 0)).sum()), total_staked=float(stake.sum()),
        gross_return=float(gross.sum()), net_profit=float(net.sum()),
        roi=float(net.sum()/stake.sum()) if stake.sum() else 0.,
        max_daily_drawdown_dollars=float((peak-cumulative).max(initial=0)),
        max_daily_exposure=float(daily.stake.max()) if len(daily) else 0.)
    if len(cumulative):
        np.testing.assert_allclose(cumulative[-1], summary['net_profit'], atol=1e-9)
    daily['cumulative_net'] = cumulative
    return summary, daily.reset_index()


def load_inputs():
    predictions = pd.read_csv(DATA / 'predictions.csv')
    source = {s: pd.read_csv(DATA / (s+'.csv')).set_index('source_row') for s in ('train','test')}
    audit = json.loads((DATA / 'data_audit.json').read_text())
    for split in source:
        assert hashlib.sha256((DATA / (split+'.csv')).read_bytes()).hexdigest() == audit['output_hashes'][split]
    return predictions, source


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    predictions, sources = load_inputs()
    files = [DATA/'predictions.csv', DATA/'train.csv', DATA/'test.csv',
             DATA/'selection.json', DATA/'selection_plan.json', DATA/'minimum_cv_audit.json',
             Path(__file__), ROOT/'src/live_bets.py', ROOT/'src/stake_levels.py']
    plan = dict(strategies=RULES, staking_modes=MODES,
        selection='Use saved model/window/feature choices. Replay all 15 existing strategies unchanged. '
                  'The filtered legacy candidate (odds 4–1000, expected return >2%) is applied '
                  'unchanged to every model, as in previous regression backtests. No strategy tuning.',
        staking='Original-rule flat: $1 for every original Bet action. Positive-edge flat: $1 only '
                'when existing tier sizing permits a bet. Tiers: existing fixed $1–$5 amounts, '
                'ceil(100*positive Kelly score), capped at level five. The latter two use identical bets.',
        accounting='Net = gross return minus stakes; ROI = net / total staked. Synthetic double chance '
                   'splits a single total stake across 1X2 outcomes with equal payout. '
                   'Daily drawdown measured from cumulative net including initial zero.',
        capital='Fixed stakes, no compounding, bankroll cap, fees, taxes or minimum-stake rounding. '
                'Max daily exposure and drawdown are dollar amounts, not bankroll percentages. '
                'Each strategy is a separate counterfactual; never add strategy profits together.',
        limits='Historical Bet365 prices, not live bets or Winner prices. 2021-22 audit and 2022-23 '
               'test have been previously inspected. Best-in-table results are hindsight rankings, '
               'not independently selected profitable strategies. Existing EA/odds timestamp limitations remain.',
        input_hashes={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (OUT/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    summaries, curves = [], []
    ledger_fields = ['family','variant','split','source_row','date','league','home','away',
                     'strategy','outcome','B365H','B365D','B365A','p_home','p_draw','p_away',
                     'home_weight','draw_weight','away_weight','unit_return',
                     'original_stake','eligible_stake','tier_stake',
                     'original_net','eligible_net','tier_net']
    ledger_count = 0
    with gzip.open(OUT/'ledger.csv.gz','wt',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=ledger_fields)
        writer.writeheader()
        for (family, variant, split), group in predictions.groupby(['family','variant','split'], sort=True):
            expected = sources['test'] if split == 'test_2022' else sources['train'].query('season == 2021')
            if not group.source_row.is_unique or set(group.source_row) != set(expected.index):
                raise ValueError('Incomplete/duplicate betting sample')
            frame = expected.loc[group.source_row].reset_index()
            if frame.fixture_date.tolist() != group.date.tolist() or frame.league.tolist() != group.league.tolist():
                raise ValueError('Fixture identity mismatch')
            outcomes = (1-np.sign(frame.home_score-frame.away_score)).to_numpy(int)
            np.testing.assert_array_equal(outcomes, group.outcome.to_numpy())
            p = group[['p_home','p_draw','p_away']].to_numpy()
            a = settle_arrays(frame, p)
            print(f'{split}: {family} / {variant} / {len(frame)} matches', flush=True)
            meta = dict(family=family, variant=variant, split=split)
            for j, strategy in enumerate(RULES):
                for league in ['ALL', *sorted(frame.league.unique())]:
                    mask = np.ones(len(frame), bool) if league == 'ALL' else (frame.league == league).to_numpy()
                    for mode, key in zip(MODES, ['original_stake','eligible_stake','tier_stake']):
                        stat, daily = summarize(frame.fixture_date.to_numpy()[mask],
                                                a[key][mask,j], a['unit_return'][mask,j])
                        info = dict(**meta, strategy=strategy, league=league, staking=mode)
                        summaries.append(dict(**info, **stat, unavailable=int(a['unavailable'][mask,j].sum())))
                        # Curves only for the combined portfolio; league totals remain in summary.
                        if league == 'ALL':
                            curves.extend(dict(**info, **r) for r in daily.to_dict('records'))
                for i, row in enumerate(frame.itertuples()):
                    if not a['original_stake'][i,j]:
                        continue
                    entry = dict(**meta, source_row=int(row.source_row), date=row.fixture_date,
                        league=row.league, home=row.home_team_name, away=row.away_team_name,
                        strategy=strategy, outcome=int(outcomes[i]),
                        B365H=row.B365H, B365D=row.B365D, B365A=row.B365A,
                        p_home=p[i,0], p_draw=p[i,1], p_away=p[i,2])
                    entry.update({k:float(a[k][i,j]) for k in
                                  ['home_weight','draw_weight','away_weight','unit_return',
                                   'original_stake','eligible_stake','tier_stake']})
                    entry.update({label+'_net':entry[key]*(entry['unit_return']-1) for label,key in
                                  [('original','original_stake'),('eligible','eligible_stake'),('tier','tier_stake')]})
                    writer.writerow(entry)
                    ledger_count += 1
    table = pd.DataFrame(summaries)
    table.to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'daily_curves.csv.gz',index=False,compression='gzip')
    primary = table[(table.split=='test_2022') & (table.league=='ALL')]
    # Side-by-side minimum-CV choices, with no outcome-based model selection.
    matched = primary[primary.variant.isin(['minimum_cv_without_form','minimum_cv_form_allowed'])]
    index = ['family','strategy','staking']
    no = matched[matched.variant=='minimum_cv_without_form'].set_index(index)
    yes = matched[matched.variant=='minimum_cv_form_allowed'].set_index(index)
    pairs = no[['net_profit','roi','bets','total_staked','max_daily_drawdown_dollars']].join(
        yes[['net_profit','roi','bets','total_staked','max_daily_drawdown_dollars']],lsuffix='_without',rsuffix='_form')
    pairs['net_change_with_form'] = pairs.net_profit_form-pairs.net_profit_without
    pairs.reset_index().to_csv(OUT/'form_profit_comparison.csv',index=False)
    best = primary.sort_values('net_profit',ascending=False).groupby(['strategy','staking'],sort=False).head(1)
    best.to_csv(OUT/'hindsight_best_by_strategy.csv',index=False)
    fixed = pairs.reset_index()
    fixed = fixed[(fixed.strategy==SEARCH_CANDIDATE)&(fixed.staking==MODES[2])]
    fixed.to_csv(OUT/'fixed_filtered_strategy.csv',index=False)

    def pretty(frame):
        return frame.to_html(index=False,float_format=lambda x:f'{x:,.3f}',escape=True)

    body = '<h1>Form models: historical betting results</h1>'
    body += '<p>Same 1,445 test fixtures (2022–23), 1,445 forward-audit fixtures (2021–22), seven model families; all saved feature-selection variants. Dollar amounts use the stated stake sizes.</p>'
    body += ''.join('<p>'+html.escape(plan[k])+'</p>' for k in ['selection','staking','accounting','capital','limits'])
    body += '<h2>Fixed filtered legacy strategy, existing $1–$5 tiers</h2><p>Minimum-CV models, with/without form. This is the previously used strategy, not selected as the winner of this rerun.</p>'
    body += pretty(fixed[['family','net_profit_without','net_profit_form','net_change_with_form',
                         'bets_without','bets_form','total_staked_form','roi_form','max_daily_drawdown_dollars_form']])
    with_form = primary[primary.variant=='minimum_cv_form_allowed']
    for mode in MODES:
        body += '<h2>'+html.escape(mode)+' — minimum-CV form-allowed models</h2>'
        grid = with_form[with_form.staking==mode].pivot(index='strategy',columns='family',values='net_profit')
        body += grid.to_html(float_format=lambda x:f'${x:,.2f}')
    body += '<h2>Largest historical profits (hindsight ranking)</h2>'
    body += pretty(primary[primary.staking==MODES[2]].sort_values('net_profit',ascending=False).head(15)[
        ['family','variant','strategy','bets','total_staked','net_profit','roi','max_daily_drawdown_dollars']])
    body += '<h2>Matched form changes across every strategy</h2>' + pretty(pairs.reset_index())
    audit = table[(table.split=='audit_2021')&(table.league=='ALL')&
                  (table.variant=='minimum_cv_form_allowed')&(table.staking==MODES[2])]
    body += '<h2>Forward audit 2021–22: form-allowed models, $1–$5 tiers</h2>'
    body += audit.pivot(index='strategy',columns='family',values='net_profit').to_html(float_format=lambda x:f'${x:,.2f}')
    body += '<details><summary>All models, variants, strategies, stake sizes and leagues</summary>'+pretty(table)+'</details>'
    (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Form model betting backtest</title>'
        '<style>body{font:15px system-ui;margin:30px;color:#182532}table{border-collapse:collapse;font-size:12px}'
        'td,th{padding:6px;border-bottom:1px solid #ddd}h2{margin-top:30px}</style>'+body,encoding='utf-8')
    (OUT/'run_summary.json').write_text(json.dumps(dict(summary_rows=len(table),ledger_rows=ledger_count,
        strategies=len(RULES),prediction_groups=predictions.groupby(['family','variant','split']).ngroups,
        hindsight_top_tier=primary[primary.staking==MODES[2]].sort_values('net_profit',ascending=False).head(10).to_dict('records')),
        indent=2),encoding='utf-8')
    print('Fixed filtered strategy:\n'+fixed.to_string(index=False),flush=True)
    print('Hindsight tier leaders:\n'+primary[primary.staking==MODES[2]].sort_values(
        'net_profit',ascending=False).head(10).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
