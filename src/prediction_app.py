"""Local browser interface for the live prediction pipeline."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import secrets
import threading
from urllib.parse import urlparse

from live_predictions import LEAGUES, LIVE, pipeline
from live_bets import enrich, export_csv
from pathlib import Path

PAGE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Football Forecast</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#0c1420;color:#e5edf6}body{max-width:1200px;margin:40px auto;padding:0 22px}
h1{font-size:38px;margin-bottom:8px}p{color:#aebed0;line-height:1.6}a{color:#73dab9}.panel{background:#142132;border:1px solid #29394d;border-radius:16px;padding:22px;margin:22px 0}
.controls{display:flex;gap:20px;flex-wrap:wrap;align-items:end}label{display:block;font-size:14px;margin:6px 0}select,input,button{font:inherit;padding:10px;border:1px solid #40546b;border-radius:8px;background:#0c1420;color:inherit}input[type=number]{width:70px}
button{background:#77e2bc;color:#0c1420;font-weight:700;cursor:pointer}button:disabled{opacity:.5;cursor:wait}.muted{font-size:13px;color:#aebed0}#status{white-space:pre-wrap}#warnings{color:#ffd391}
.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px}th{text-align:left;color:#aebed0;font-size:12px;text-transform:uppercase}td,th{padding:14px 10px;border-bottom:1px solid #29394d;white-space:nowrap}.prob{color:#77e2bc;font-variant-numeric:tabular-nums}
.metrics{display:flex;gap:12px;flex-wrap:wrap}.metric{background:#0c1420;border-radius:10px;padding:14px;flex:1;min-width:150px}.metric b{display:block;margin-bottom:8px}#download{display:inline-block;margin:14px 0}summary{cursor:pointer}
#bets td:first-child{white-space:normal;min-width:170px} #bets td:nth-child(2){white-space:normal;min-width:160px}
[hidden]{display:none!important}
</style><h1>Football Forecast</h1><p>Collect match data → build model inputs → predict scheduled matches.</p>
<div class="panel"><div class="controls"><div><label for="model">Model</label><select id="model"><option value="lineup_rf">Lineups + EA ratings (saved RF)</option><option value="poisson">Results-only Poisson baseline</option></select></div><div><label for="league">League</label><select id="league"><option value="all">All five leagues</option>__LEAGUES__</select></div>
<div><label for="seasons">History (seasons)</label><input id="seasons" type="number" min="1" max="10" value="3"></div>
<div><label for="days">Next days</label><input id="days" type="number" min="1" max="90" value="14"></div>
<label><input type="checkbox" id="offline"> Use cached files</label><button id="run">Refresh & predict</button></div>
<p id="status" role="status" aria-live="polite">Ready. Refresh to collect data and generate predictions.</p><div id="warnings" role="alert"></div></div>
<div class="panel" id="manual-lab"><h2>Manual strategy filters</h2><p>Set your own filters, then apply to update selections and historical profit curves using saved Bet365 odds. Outcome and odds filters apply to the original strategy's chosen selection. Expected return must be strictly above your minimum; odds limits are inclusive. Flat staking uses the level-1 amount.</p><div id="manual-controls" class="controls"></div><div id="manual-quotes"></div><p id="manual-status" role="status">Choose filters and apply.</p><div id="manual-output"></div></div>
<div class="panel"><h2>Saved strategy presets</h2><label for="strategy">Compare a betting rule</label><select id="strategy"><option value="Legacy variance — filtered search candidate">Legacy variance — filtered search candidate</option></select> <a href="/strategies.csv">Download all strategy selections</a>
<p class="muted">Five-level staking skips selections with no positive model edge. Kelly score = model EV / (decimal odds − 1). Levels 1–5 cover scores (0,1%], (1%,2%], (2%,3%], (3%,4%], and above 4%. These fixed tiers rank stakes; they are not bankroll percentages or a guarantee of profit.</p>
<div class="controls" id="stake-inputs"></div><p class="muted" id="stake-message">Dollar amounts apply to the total stake per match. Edits update this view only; CSV exports use $1–$5. Synthetic double chance splits that total across two bets. Saved odds may have changed.</p>
<div class="scroll"><table><thead><tr><th>Match</th><th id='strategy-heading'>Strategy</th><th>Selection / staking action</th><th>Odds</th><th>Probability</th><th>Model EV / unit</th><th>Dollar stake split</th><th>Level / total</th></tr></thead><tbody id="bets"></tbody></table></div></div>
<div class="panel"><h2>Historical dollar returns</h2><p class="muted">Uses the saved lineup vector RF and the betting rule selected above. Only held-out test results are shown. Training-period results require a chronological backtest. Changing the live prediction model does not change this historical model.</p><p id="stake-method" class="muted">Loading historical results…</p><p><a href="/stake-results.csv">Download season and level results ($1–$5)</a> · <a href="/stake-ledger.csv">Download every historical bet ($1–$5)</a></p><h3>By season</h3><div class="scroll"><table><thead><tr><th>Strategy</th><th>Split</th><th>Season</th><th>Bets / skips / unavailable</th><th>Total staked</th><th>Gross returns</th><th>Net profit</th><th>ROI</th><th>Original flat $1 net</th></tr></thead><tbody id="stake-seasons"></tbody></table></div><h3>By stake level · all seasons in each split</h3><div class="scroll"><table><thead><tr><th>Strategy</th><th>Split</th><th>Level</th><th>Bets</th><th>Total staked</th><th>Gross returns</th><th>Net profit</th><th>ROI</th></tr></thead><tbody id="stake-levels"></tbody></table></div></div>
<div class="panel"><h2>Upcoming matches</h2><p id="stamp" class="muted">No run loaded.</p><a id="download" href="/predictions.csv" hidden>Download predictions CSV</a>
<div class="scroll"><table><thead><tr><th>Date / source time</th><th>League</th><th>Match</th><th>Home</th><th>Draw</th><th>Away</th><th>Expected goals</th><th>Likely score</th></tr></thead><tbody id="rows"></tbody></table></div></div>
<div class="panel" id="unavailable" hidden><h2>Fixtures needing attention</h2><p class="muted">These fixtures have no lineup-model prediction. Resolve the missing inputs, then refresh.</p><div id="skipped"></div></div>
<div class="panel"><h2 id="validation-title">Model validation</h2><p class="muted" id="validation-note"></p><div id="metrics" class="metrics"></div></div>
<details class="panel"><summary>Model & data details</summary><p>The lineup model downloads RotoWire expected/confirmed starting XIs and current EA overall ratings, builds rating summaries, season totals and last-five-match form, and uses Bet365 odds from Football-Data. Those inputs feed the existing 2,000-tree vector random forest. Missing or ambiguous ratings withhold the fixture; they never trigger an automatic baseline substitution. Expand a match to see its 22 players and ratings.</p><p>Matchweek is estimated only when both clubs have played the same number of games, or supplied through data/live_overrides.json. Estimates require verification after postponements. Unknown teams use the trained encoder's unknown-category handling and are flagged. The saved forest has not been validated on current predicted lineups or EA launch ratings.</p><p>The optional results-only baseline uses home/away attack and defence, a 180-day recency half-life, and league-average shrinkage. Both models convert predicted goal counts into score probabilities with independent Poisson distributions. Expected goals here are predicted goal counts, not shot-based xG.</p>
<p>Results and the provider’s limited fixture list come from <a href="https://www.football-data.co.uk/data.php" target="_blank" rel="noreferrer">Football-Data.co.uk</a>. Match dates are filtered by the current UTC date; times are shown exactly as provided. Today's listed fixtures may already have started. Training excludes all results dated today. Probabilities are model estimates; validation is diagnostic, not a profit guarantee.</p><pre id="sources" class="muted" style="white-space:pre-wrap"></pre></details>
<script>
const $=id=>document.getElementById(id), names=__NAMES__;let displayed=null,currentReport=null;
function cell(tr,text,cls){const td=document.createElement('td');td.textContent=text;if(cls)td.className=cls;tr.append(td)}
function renderBets(){
const showAll=$('strategy').value==='all';
$('strategy-heading').hidden=!showAll;
const body=$('bets');body.replaceChildren();if(!currentReport)return;
for(const p of currentReport.predictions){for(const b of p.strategies??[]){
if(!showAll&&b.strategy!==$('strategy').value)continue;
const row=document.createElement('tr');cell(row,`${p.date} · ${p.home} vs ${p.away}`);
cell(row,b.strategy);row.lastElementChild.hidden=!showAll;
cell(row,b.stake_level?b.selection:(b.staking_action??b.action)+(b.action==='Bet'?' · '+b.selection:''),b.stake_level?'prob':'');
cell(row,b.decimal_odds===null?'—':b.decimal_odds.toFixed(2)+(b.synthetic?' (synthetic)':''));
cell(row,b.probability===null?'—':(100*b.probability).toFixed(1)+'%');
cell(row,b.expected_net_per_unit===null?'—':(b.expected_net_per_unit>=0?'+':'')+b.expected_net_per_unit.toFixed(3));
const amount=b.stake_level?(typeof stakeAmounts==='undefined'?[1,2,3,4,5]:stakeAmounts)[b.stake_level-1]:0;
cell(row,amount?[['Home',b.home_stake],['Draw',b.draw_stake],['Away',b.away_stake]].filter(x=>x[1]>0).map(x=>`${x[0]}: $${(amount*x[1]).toFixed(2)}`).join(' + '):'$0');
cell(row,b.stake_level?`${b.stake_level} / $${amount.toFixed(2)}`:'Skip / $0');body.append(row);
const note=[b.note,b.staking_reason].filter(Boolean).join(' ');
if(note){const noteRow=document.createElement('tr'),td=document.createElement('td');td.colSpan=showAll?8:7;td.className='muted';td.style.whiteSpace='normal';td.textContent=note;noteRow.append(td);body.append(noteRow)}
}}
if(!body.children.length){const row=document.createElement('tr');cell(row,'No predictions with strategy selections available.');body.append(row)}
if(typeof renderStakeHistory==='function')renderStakeHistory();
}
$('strategy').onchange=()=>{renderBets();renderStakeHistory()};
function render(r){$('stamp').textContent=`Generated ${r.generated_at} · ${r.model_kind==='lineup_rf'?'Current-season form':r.seasons+' seasons'} · ${r.days}-day window · ${r.model} · ${r.predictions.length} predictions`;
currentReport=r;const selected=$('strategy').value||'Square-root profit weighting';$('strategy').replaceChildren();for(const name of [...(r.strategy_names??[]),'all']){const option=document.createElement('option');option.value=name;option.textContent=name==='all'?'All strategies':name;$('strategy').append(option)}$('strategy').value=selected;renderBets();
if(typeof manualSetReport==='function')manualSetReport(r);
$('warnings').textContent=r.warnings.join('\n');$('rows').replaceChildren();
for(const p of r.predictions){const tr=document.createElement('tr');cell(tr,`${p.date} ${p.time}`);cell(tr,names[p.league]);cell(tr,`${p.home} vs ${p.away}${p.limited_history?' *':''}`);for(const key of ['p_home','p_draw','p_away'])cell(tr,(p[key]*100).toFixed(1)+'%','prob');cell(tr,`${p.expected_home_goals.toFixed(2)} – ${p.expected_away_goals.toFixed(2)}`);cell(tr,p.modal_score);$('rows').append(tr);if(p.home_lineup){const row=document.createElement('tr'),td=document.createElement('td'),details=document.createElement('details'),summary=document.createElement('summary'),body=document.createElement('div');td.colSpan=8;td.style.whiteSpace='normal';summary.textContent=`Show lineups & ratings · ${p.lineup_status} · ${p.ratings_edition} · Matchweek ${p.matchweek}`;body.style.whiteSpace='pre-wrap';body.textContent=['HOME: '+p.home_lineup.map(x=>`${x.lineup_name}: ${x.overall}${x.injury_flag?' ('+x.injury_flag+')':''}`).join(' · '),'AWAY: '+p.away_lineup.map(x=>`${x.lineup_name}: ${x.overall}${x.injury_flag?' ('+x.injury_flag+')':''}`).join(' · '),p.notes].join('\n\n');details.append(summary,body);td.append(details);row.append(td);$('rows').append(row)}}
if(!r.predictions.length){const tr=document.createElement('tr');cell(tr,'No predictions available. See warnings and missing inputs below.');$('rows').append(tr)}
$('unavailable').hidden=!(r.skipped?.length);$('skipped').replaceChildren();for(const f of r.skipped??[]){const p=document.createElement('p');p.textContent=`${f.date} · ${f.home} vs ${f.away}: ${f.reason}`;$('skipped').append(p)}
$('validation-note').textContent=r.model_kind==='lineup_rf'?'Using the existing historical random forest. No live-lineup accuracy claim is available. See the saved experiment report for its historical test results.':'Up to 60 recent matches per league, predicted using only earlier dates. Lower log loss is better; compare against league-average goals.';
$('metrics').replaceChildren();for(const [key,m] of Object.entries(r.validation)){const box=document.createElement('div');box.className='metric';const title=document.createElement('b');title.textContent=names[key];box.append(title);const text=document.createElement('div');text.textContent=`${m.matches} evaluated · Accuracy ${m.accuracy===null?'n/a':(m.accuracy*100).toFixed(1)+'%'} · Log loss ${m.log_loss?.toFixed(3)??'n/a'} · Baseline ${m.league_baseline_log_loss?.toFixed(3)??'n/a'} · ${m.training_matches} training matches · Latest result ${m.latest_result}`;box.append(text);$('metrics').append(box)}
$('sources').textContent=(r.model_kind==='lineup_rf'?'* Team unseen in historical training.':'* Limited recent team history; estimates lean on league averages.')+'\n\n'+r.sources.map(s=>`${s.url}\nFetched ${s.fetched_at}${s.cached?' (cached)':''}`).join('\n\n');$('download').hidden=false;}
async function poll(){try{const res=await fetch('/api/status');if(!res.ok)throw Error('Unable to read server status');const s=await res.json();$('run').disabled=s.running;$('status').textContent=s.error?'Run failed: '+s.error:s.message;if(s.result&&s.result.generated_at!==displayed){render(s.result);displayed=s.result.generated_at}if(s.error&&s.result)$('status').textContent+='\nShowing the previous successful run below.'}catch(e){$('status').textContent=e.message}finally{setTimeout(poll,1200)}}
$('model').onchange=()=>{$('seasons').disabled=$('model').value==='lineup_rf'};$('model').onchange();
$('run').onclick=async()=>{const seasons=Number($('seasons').value),days=Number($('days').value);if(!Number.isInteger(seasons)||seasons<1||seasons>10||!Number.isInteger(days)||days<1||days>90){$('status').textContent='Choose 1–10 seasons and 1–90 days.';return}$('run').disabled=true;try{const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json','X-App-Token':'__TOKEN__'},body:JSON.stringify({model:$('model').value,leagues:$('league').value==='all'?Object.keys(names):[$('league').value],seasons,days,offline:$('offline').checked})});if(!r.ok)throw Error((await r.json()).error);$('status').textContent='Starting…'}catch(e){$('status').textContent=e.message;$('run').disabled=false}};poll();
</script><script src="/stake-ui.js"></script><script src="/manual-ui.js" data-token="__TOKEN__"></script></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args(argv)
    token = secrets.token_urlsafe(24)
    state = {'running': False, 'message': 'Ready. Refresh to generate predictions.', 'error': None, 'result': None}
    lock = threading.Lock()
    saved = LIVE / 'predictions.json'
    if saved.exists():
        try:
            state['result'] = enrich(json.loads(saved.read_text(encoding='utf-8')))
            state['message'] = 'Showing the previous successful run. Refresh for updated data.'
        except (ValueError, OSError):
            pass
    page = PAGE.replace('__LEAGUES__', ''.join(f'<option value="{k}">{v}</option>' for k, v in LEAGUES.items()))
    page = page.replace('__NAMES__', json.dumps(LEAGUES)).replace('__TOKEN__', token).encode()

    def work(options):
        def progress(message):
            with lock:
                state['message'] = message
        try:
            result = pipeline(**options, progress=progress)
            with lock:
                state['result'] = result
        except Exception as exc:
            with lock:
                state['error'] = str(exc)
        finally:
            with lock:
                state['running'] = False

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, payload, content_type='application/json'):
            if not isinstance(payload, bytes):
                payload = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type + '; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.end_headers()
            self.wfile.write(payload)

        def valid_host(self):
            return self.headers.get('Host') in {f'127.0.0.1:{args.port}', f'localhost:{args.port}'}

        def do_GET(self):
            if not self.valid_host():
                return self.send(403, {'error': 'Invalid host'})
            path = urlparse(self.path).path
            if path == '/':
                return self.send(200, page, 'text/html')
            if path == '/stake-ui.js':
                return self.send(200, Path(__file__).with_name('stake-ui.js').read_bytes(), 'application/javascript')
            if path == '/manual-ui.js':
                return self.send(200, Path(__file__).with_name('manual-ui.js').read_bytes(), 'application/javascript')
            stake_files = {'/api/stake-results': ('stake_level_results.json','application/json'),
                           '/strategy-search': ('strategy_search.html','text/html'),
                           '/strategy-search.csv': ('strategy_search_finalists.csv','text/csv'),
                           '/api/stake-curves': ('stake_cumulative.json','application/json'),
                           '/stake-results.csv': ('stake_level_results.csv','text/csv'),
                           '/stake-ledger.csv': ('stake_level_bet_ledger.csv','text/csv')}
            if path in ('/strategy-search','/strategy-search.csv'):
                return self.send(410, {'error': 'The earlier OOB-based strategy search is retired. Use manual filters with held-out test results.'})
            if path in stake_files:
                filename, content_type = stake_files[path]
                artifact = LIVE.parent / filename
                if artifact.exists():
                    return self.send(200, artifact.read_bytes(), content_type)
                return self.send(404, {'error':'Run python scripts/backtest_stake_levels.py to build historical results.'})
            if path == '/api/status':
                with lock:
                    if not state['running'] and saved.exists():
                        try:
                            latest = json.loads(saved.read_text(encoding='utf-8'))
                            if latest.get('generated_at') != (state['result'] or {}).get('generated_at'):
                                state['result'] = enrich(latest)
                                state['message'] = 'Loaded the latest saved prediction run.'
                        except (ValueError, OSError):
                            pass
                    return self.send(200, state)
            if path == '/predictions.csv' and (LIVE / 'predictions.csv').exists():
                return self.send(200, (LIVE / 'predictions.csv').read_bytes(), 'text/csv')
            if path == '/strategies.csv':
                with lock:
                    if state['result']:
                        return self.send(200, export_csv(state['result']).encode(), 'text/csv')
            self.send(404, {'error': 'Not found'})

        def do_POST(self):
            if not self.valid_host() or self.headers.get('X-App-Token') != token:
                return self.send(403, {'error': 'Invalid request token or host'})
            if self.path == '/api/manual':
                try:
                    length=int(self.headers.get('Content-Length',0))
                    if not 0<length<=65536:raise ValueError('Invalid request size')
                    options=json.loads(self.rfile.read(length))
                    if not isinstance(options,dict):raise ValueError('Expected an object')
                    if self.path=='/api/poly-event':
                        from odds_sources import discover
                        result=discover(options.get('reference',''))
                    else:
                        with lock:
                            report=json.loads(json.dumps(state['result']))
                        if not report:raise ValueError('Generate or load predictions first')
                        if self.path=='/api/manual':
                            if options.get('source','bet365')!='bet365':raise ValueError('Manual interface currently uses saved Bet365 odds only')
                            from manual_strategies import evaluate
                            result=evaluate(report,options)
                        else:
                            from odds_sources import fetch_prices,fixture_key
                            fixture=next((m for m in report['predictions'] if fixture_key(m)==options.get('fixture')),None)
                            if not fixture:raise ValueError('Fixture is no longer in the saved report')
                            result=fetch_prices(options.get('event'),options.get('markets',{}),
                                {k:fixture[k] for k in ('date','home','away')},options.get('confirmed'))
                    return self.send(200,result)
                except (ValueError,KeyError,TypeError,OSError) as exc:
                    return self.send(400,{'error':str(exc)})
            if self.path != '/api/run':
                return self.send(404, {'error': 'Not found'})
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 4096:
                    raise ValueError('Invalid request length')
                options = json.loads(self.rfile.read(length))
                if set(options) != {'leagues', 'seasons', 'days', 'offline', 'model'}:
                    raise ValueError('Invalid options')
                if (not isinstance(options['leagues'], list) or not options['leagues']
                        or any(not isinstance(l, str) or l not in LEAGUES for l in options['leagues'])
                        or type(options['seasons']) is not int or not 1 <= options['seasons'] <= 10
                        or type(options['days']) is not int or not 1 <= options['days'] <= 90
                        or type(options['offline']) is not bool
                        or options['model'] not in ('poisson', 'lineup_rf')):
                    raise ValueError('Invalid league, seasons, days, or cache option')
            except (ValueError, TypeError) as exc:
                return self.send(400, {'error': str(exc)})
            with lock:
                if state['running']:
                    return self.send(409, {'error': 'A prediction run is already active'})
                state.update(running=True, message='Starting…', error=None)
            threading.Thread(target=work, args=(options,), daemon=True).start()
            self.send(202, {'started': True})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Football Forecast: http://127.0.0.1:{args.port}  (Ctrl+C to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
