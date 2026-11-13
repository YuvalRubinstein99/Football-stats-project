"""Readable report of the frozen regularization experiment; no model changes."""
import html
import json
import hashlib
from select_expanded_features import pd
from train_regularized_home_win import OUT
from prepare_regularized_holdout import DEST
from plot_binary_training_oos import render


def main():
    frozen=json.loads((OUT/'frozen_selection.json').read_text())
    plan=json.loads((OUT/'plan.json').read_text())
    audit=json.loads((DEST/'data_audit.json').read_text())
    evaluation=json.loads((DEST/'evaluation.json').read_text())
    assert hashlib.sha256((OUT/'frozen_model.joblib').read_bytes()).hexdigest()==frozen['model_sha256']
    fitted_ledger=pd.read_csv(OUT/'in_sample_ledger.csv.gz')
    fitted_summary=pd.read_csv(OUT/'in_sample_summary.csv')
    assert fitted_ledger.match_id.nunique()==frozen['training_matches']
    assert set(fitted_ledger.model)=={frozen['model']}
    # Original source_row values restart in the appended 2022 season.
    fitted_ledger['source_row']=fitted_ledger.match_id
    curves=render(fitted_ledger,fitted_summary,'in_sample',out=OUT,
        heading='Regularized model: in-sample cumulative profit',
        subtitle='Same frozen depth-2 forest predicts its own training rows | 2014-15 to 2022-23',
        note='Home-only EV >0% (teal) and >2% (purple). Ordinary fitted-model predictions; no refitting.',
        footer='IN-SAMPLE: training fit. Compare ROI, not total profit, with the smaller holdout. Different vertical scales.')
    pd.DataFrame(curves).to_csv(OUT/'in_sample_daily_curves.csv',index=False)
    comparison=fitted_summary[['strategy','staking','matches','bets','net','roi']].merge(
        pd.read_csv(DEST/'summary.csv')[['strategy','staking','matches','bets','net','roi']],
        on=['strategy','staking'],validate='one_to_one',suffixes=('_train','_test'))
    comparison['roi_gap_percentage_points']=(comparison.roi_train-comparison.roi_test)*100
    comparison.to_csv(OUT/'train_test_comparison.csv',index=False)
    render(pd.read_csv(DEST/'ledger.csv.gz'),pd.read_csv(DEST/'summary.csv'),'holdout_2025',out=DEST,
        heading='Frozen regularized model: 2025-26 holdout',
        subtitle='Depth-2 forest, minimum 100 samples per leaf | Model and thresholds frozen before evaluation',
        note='Home-only EV >0% (teal, selected cutoff) and >2% (purple, fixed comparison).',
        footer='Held out from this experiment. Dated FC23/FC24 ratings are stale; eligible lineup subset. Different vertical scales.')
    body='<h1>Regularization and chronological calibration</h1>'
    body+='<p><strong>The frozen candidate did not demonstrate a betting edge on the held-out 2025-26 sample.</strong> '
    body+='It also had slightly worse log loss than normalized bookmaker probabilities on those same matches.</p>'
    body+='<p>Model: '+html.escape(frozen['model'])+'. Training ended '+frozen['training_last_date']
    body+=f' ({frozen["training_matches"]:,} matches). Selected home-only expected-return cutoff: {frozen["threshold"]:.0%}.</p>'
    body+='<p>The final choice is a 500-tree forest with maximum depth 2, minimum leaf size 100, '
    body+='and 12 compact inputs. Team names and redundant rating transformations are omitted. '
    body+='Sigmoid calibration was tested using past forward predictions; the final selection favored raw probabilities.</p>'
    for key in ['calibration','selection','threshold','history','final']:
        body+='<p><strong>'+key.capitalize()+': </strong>'+html.escape(plan[key])+'</p>'
    body+=f'<p><strong>Coverage:</strong> {audit["eligible"]} of {audit["matches"]} matches. '
    body+=html.escape(audit['limitation'])+'</p>'
    body+='<img src="in_sample_cumulative.svg" alt="Same frozen model in-sample cumulative profit">'
    body+='<img src="holdout_2025/holdout_2025_cumulative.svg" alt="Frozen holdout cumulative profit">'
    body+='<h2>Same model: training fit versus holdout</h2>'+comparison.to_html(index=False,float_format=lambda v:f'{v:.5f}')
    body+='<p>Training has 9,586 matches and the holdout has 286 eligible matches; compare ROI rather than '
    body+='total profit. The gap reflects possible overfitting, sampling variation and changes across seasons '
    body+='and rating coverage. The earlier model used a shorter training period, so comparing its training '
    body+='ROI with this model is not a controlled estimate of the effect of regularization.</p>'
    for path,title in [(DEST/'summary.csv','Holdout betting results'),(DEST/'metrics.csv','Holdout probability accuracy'),
                       (DEST/'coverage.csv','Holdout coverage'),(OUT/'replay_metrics.csv','Historical replay: identical fixtures'),
                       (OUT/'threshold_policy_summary.csv','Actual threshold-selection policy'),
                       (OUT/'replay_summary.csv','Historical replay: fixed-filter diagnostics'),
                       (OUT/'in_sample_summary.csv','Final candidate: in-sample fit')]:
        body+='<h2>'+title+'</h2>'+pd.read_csv(path).to_html(index=False,float_format=lambda v:f'{v:.5f}')
    body+='<p>The historical threshold-selection policy placed zero bets: none of the chosen models had '
    body+='a cutoff meeting the two-season profitability requirement at those five decision points. '
    body+='The positive fixed-filter historical results are therefore diagnostic, not the return of the '
    body+='adaptive threshold policy. The final holdout choice is different: the frozen raw depth-2 forest '
    body+='passed that requirement on 2021 and 2022, and its selected positive-EV filter was applied unchanged.</p>'
    body+='<img src="historical_replay_cumulative.svg" alt="Historical fixed-filter diagnostic">'
    body+='<p>'+html.escape(evaluation['uncertainty'])+'</p>'
    body+='<p>Five unit tests and the independent verification script check temporal isolation, frozen '
    body+='selection and inference, feature preprocessing, all saved bet settlements, and graph endpoints. '
    body+='The original model artifacts remain unchanged. This experiment is saved separately.</p>'
    (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Regularized home-win evaluation</title>'
        '<style>body{font:15px system-ui;margin:32px;color:#263548}p{max-width:1100px}img{width:100%;max-width:1200px}'
        'table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border-bottom:1px solid #ddd}</style>'+body,encoding='utf-8')
    print(OUT/'report.html')


if __name__=='__main__':main()
