"""Forward evaluation of goal margins and Bet365 full-time Asian handicaps."""
from backtest_goal_totals import OUT, forward_goal_forecasts, np, pd
from goal_difference import handicap_edges, handicap_net, margin_bands
from plot_binary_home_filters import Drawing, Rect, Line, PolyLine, HexColor, text, renderPDF, pdfium


def plot(daily):
    d = Drawing(820,310)
    d.add(Rect(0,0,820,310,fillColor=HexColor('#ffffff'),strokeColor=None))
    text(d,32,275,'Asian handicap · cumulative net profit',21,bold=True)
    text(d,32,249,'One unit per qualifying match, settled at the archived Bet365 price',12)
    dates = pd.to_datetime(daily.index)
    start = dates.min() - pd.Timedelta(days=1)
    x = lambda day: 55 + 675*(day-start).days/(dates.max()-start).days
    low, high = min(0, daily.min().min()), max(0, daily.max().max())
    y = lambda amount: 45 + 135*(amount-low)/(high-low or 1)
    for v in sorted({low,0,high}):
        d.add(Line(55,y(v),730,y(v),strokeColor=HexColor('#d3dbdb')))
        text(d,48,y(v)-3,f'{v:.0f}',9,textAnchor='end')
    for i,(column,color) in enumerate(zip(daily,['#18766b','#7953a4'])):
        values = daily[column]
        text(d,32+265*i,216,column+f'   {values.iat[-1]:+.2f} units',12,color=color,bold=True)
        points = [x(start),y(0)]
        for day, value in zip(dates, values):
            points.extend([x(day),y(value)])
        d.add(PolyLine(points,fillColor=None,strokeColor=HexColor(color),strokeWidth=1.6))
    for date in [dates.min(),pd.Timestamp('2021-08-01'),pd.Timestamp('2022-08-01'),dates.max()]:
        text(d,x(date),20,date.strftime('%b %Y'),10,textAnchor='middle')
    with pdfium.PdfDocument(renderPDF.drawToString(d)) as doc:
        page=doc[0]
        bitmap=page.render(scale=1.5)
        bitmap.to_pil().save(OUT/'goal_difference.png')
        bitmap.close()
        page.close()


def main():
    ledgers = []
    for year, n, frame in forward_goal_forecasts():
        h,a = frame.mean_home.to_numpy(), frame.mean_away.to_numpy()
        frame['mean_margin'] = h-a
        actual = (frame.hg-frame.ag).to_numpy()
        frame['actual_margin'] = actual
        bands = pd.DataFrame(margin_bands(h,a),index=frame.index)
        np.testing.assert_allclose(bands.sum(axis=1),1.)
        frame = frame.join(bands)
        line, oh, oa = (frame[k].to_numpy(float) for k in ['AHh','B365AHH','B365AHA'])
        home_ev,away_ev = handicap_edges(h,a,line,oh,oa)
        available = np.isfinite(home_ev) & np.isfinite(away_ev)
        home = home_ev >= away_ev
        frame['side'] = np.where(available,np.where(home,'Home','Away'),'')
        frame['edge'] = np.maximum(home_ev,away_ev)
        frame['unit_net'] = np.where(available,np.where(home,handicap_net(actual,line,oh),
                                                       handicap_net(-actual,-line,oa)),np.nan)
        fields = ['season','date','league','home','away','hg','ag','mean_margin','actual_margin',
                  *bands.columns,'AHh','B365AHH','B365AHA','side','edge','unit_net']
        ledgers.append(frame[fields])
        print(year,len(frame),'of',n,'matched,',int(available.sum()),'priced',flush=True)
    ledger=pd.concat(ledgers,ignore_index=True)
    ledger.to_csv(OUT/'goal_difference.csv.gz',index=False)
    print('MAE', (ledger.mean_margin-ledger.actual_margin).abs().mean(),
          'zero-margin baseline MAE',ledger.actual_margin.abs().mean(),
          'predicted/actual home advantage',ledger.mean_margin.mean(),ledger.actual_margin.mean())
    print('draw frequency',ledger.actual_margin.eq(0).mean(),'home 2+',ledger.actual_margin.ge(2).mean(),
          'away 2+',ledger.actual_margin.le(-2).mean())
    curves={}
    for cutoff,label in [(0,'Positive expected return'),(.02,'Expected return above 2%')]:
        taken=ledger[ledger.edge>cutoff]
        print(label,len(taken),taken.unit_net.sum(),taken.unit_net.mean())
        print(taken.groupby('season').unit_net.agg(['count','sum','mean']))
        curves[label]=ledger.unit_net.where(ledger.edge>cutoff,0).groupby(ledger.date).sum().cumsum()
    plot(pd.DataFrame(curves))


if __name__=='__main__':
    main()
