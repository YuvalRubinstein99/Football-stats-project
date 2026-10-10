"""Export historical five-level stake figures without refitting any model."""
from pathlib import Path
import json
from collections import defaultdict
from reportlab.graphics.shapes import Drawing, Rect, Line, String
from reportlab.graphics import renderSVG, renderPDF
from reportlab.lib.colors import HexColor
import pypdfium2 as pdfium

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'
TRAIN='#3964b1'; TEST='#009b80'


def text(d,x,y,s,size=11,color='#25354b',**kw):
    d.add(String(x,y,s,fontName='Helvetica',fontSize=size,fillColor=HexColor(color),**kw))


def bars(d,items,top,left,width,step,percent=False):
    lo=min(0,min(v for _,v,_ in items));hi=max(0,max(v for _,v,_ in items))
    padding=max(1,(hi-lo)*.18);lo-=padding;hi+=padding
    scale=lambda v:left+(v-lo)/(hi-lo)*width
    d.add(Line(scale(0),top+12,scale(0),top-step*(len(items)-1)-12,strokeColor=HexColor('#8999aa')))
    for i,(name,value,color) in enumerate(items):
        y=top-i*step
        text(d,left-12,y-4,name,10,textAnchor='end')
        d.add(Rect(min(scale(0),scale(value)),y-8,max(.7,abs(scale(value)-scale(0))),16,fillColor=HexColor(color),strokeColor=None))
        label=f'{value:+.2f}%' if percent else f'{"-" if value<0 else "+"}${abs(value):,.2f}'
        text(d,scale(value)+(5 if value>=0 else -5),y-4,label,10,textAnchor='start' if value>=0 else 'end')


def save(d,name):
    renderSVG.drawToFile(d,str(OUT/(name+'.svg')))
    with pdfium.PdfDocument(renderPDF.drawToString(d)) as doc:
        page=doc[0]; bitmap=page.render(scale=1.5)
        bitmap.to_pil().save(OUT/(name+'.png'));bitmap.close();page.close()


def main():
    report=json.loads((OUT/'stake_level_results.json').read_text(encoding='utf-8'))
    totals=defaultdict(lambda:[0.,0.])
    for s in report['rows']:
        t=totals[s['strategy'],s['split'],s['season']]
        t[0]+=s['stake_dollars'];t[1]+=s['net_profit_dollars']
    chosen='Square-root profit weighting'
    d=Drawing(1120,620);d.add(Rect(0,0,1120,620,fillColor=HexColor('#ffffff'),strokeColor=None))
    text(d,35,580,'Historical net profit by season',26)
    text(d,35,551,chosen+' | Five stake levels: $1 / $2 / $3 / $4 / $5',13)
    for i,split in enumerate(['Test']):
        stake,profit=totals[chosen,split,'All seasons']
        text(d,35+i*555,510,f'{split}: {"-" if profit<0 else "+"}${abs(profit):,.2f} net | {profit/stake*100:+.2f}% ROI',17,TRAIN if i==0 else TEST)
        text(d,35+i*555,488,f'${stake:,.2f} staked; all seasons in this split',11)
    items=[]
    for strategy,split,season in sorted(totals,key=lambda k:k[2]):
        if strategy==chosen and season!='All seasons':
            items.append((season.replace('–','-')+'  '+split,totals[strategy,split,season][1],TRAIN if split.startswith('Train') else TEST))
    bars(d,items,445,260,735,38)
    text(d,35,64,'Held-out test only (2022-23). Training-period reporting is disabled.',11)
    text(d,35,44,'Net profit subtracts stakes. Training results require a chronological backtest. Historical actual lineups differ from live expected lineups.',10)
    text(d,35,24,'Fixed staking thresholds; no compounding, fees or bankroll limit. Hypothetical returns, not guaranteed future winnings.',10)
    save(d,'stake_history_seasons')
    d=Drawing(1250,1130);d.add(Rect(0,0,1250,1130,fillColor=HexColor('#ffffff'),strokeColor=None))
    text(d,35,1090,'Held-out test ROI across strategies',25)
    text(d,35,1063,'Five-level $1-$5 stakes with positive-edge gate | ROI = net profit / total amount staked',12)
    text(d,35,1037,'Held-out test, 2022-23. Training-period reporting is disabled.',12)
    items=[]
    for strategy in sorted({k[0] for k in totals}):
        for split in ['Test']:
            stake,profit=totals[strategy,split,'All seasons']
            items.append((strategy+' / '+split,100*profit/stake if stake else 0,TRAIN if split.startswith('Train') else TEST))
    bars(d,items,994,440,665,32,percent=True)
    text(d,35,60,'Training-period profit is unavailable pending a chronological backtest.',11)
    text(d,35,39,'The test set has been examined in earlier project analyses. Comparing strategies here does not establish a future winner.',11)
    save(d,'stake_history_strategies')
    print('Saved stake_history_seasons and stake_history_strategies (PNG/SVG).')


if __name__=='__main__':main()
