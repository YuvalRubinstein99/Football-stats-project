"""Daily reinvestment of filtered-variance picks from forward goal forecasts."""
from backtest_goal_totals import forward_goal_forecasts, OUT, pd, np
from backtest_xgboost_goal_vectors import outcome_probabilities
from backtest_transfermarkt_strategies import selections, SEARCH_CANDIDATE
from plot_binary_home_filters import Drawing, Rect, Line, PolyLine, HexColor, text, renderPDF, pdfium


def main():
    parts=[]
    for year, _, frame in forward_goal_forecasts():
        p=outcome_probabilities(frame[['mean_home','mean_away']].to_numpy())
        bets=selections(frame,p)[SEARCH_CANDIDATE]
        part=frame[['date','season','home_team_name','away_team_name']].copy()
        part['bet']=bets[:,0]
        part['unit_net']=bets[:,4]
        parts.append(part[part.bet==1])
    bets=pd.concat(parts).sort_values('date')
    bets.to_csv(OUT/'rolling_outcome_bets.csv',index=False)
    summaries,paths=[],[]
    methods=[('Flat 1 unit',None,'#18766b'),('0.1% of balance',.001,'#3067a1'),
             ('0.5% of balance',.005,'#7953a4'),('1% of balance',.01,'#be6236')]
    for name,fraction,_ in methods:
        balance=peak=1000.
        drawdown=spent=0.
        for date,group in bets.groupby('date',sort=True):
            stake=1. if fraction is None else balance*fraction
            stake=min(stake,balance/len(group))
            spent+=stake*len(group)
            balance+=stake*group.unit_net.sum()
            peak=max(peak,balance)
            drawdown=max(drawdown,1-balance/peak)
            paths.append(dict(method=name,date=date,balance=balance))
        summaries.append(dict(method=name,bets=len(bets),ending_balance=balance,
                              net=balance-1000,staked=spent,roi=(balance-1000)/spent,max_drawdown=drawdown))
    summary=pd.DataFrame(summaries)
    curves=pd.DataFrame(paths)
    np.testing.assert_allclose(summary.iloc[0].net,bets.unit_net.sum())
    summary.to_csv(OUT/'rolling_outcome_summary.csv',index=False)
    curves.to_csv(OUT/'rolling_outcome_curves.csv',index=False)
    print(summary.to_string(index=False))
    print(bets.groupby('season').unit_net.agg(['count','sum']))
    d=Drawing(1000,470)
    d.add(Rect(0,0,1000,470,fillColor=HexColor('#ffffff'),strokeColor=None))
    text(d,35,428,'Outcome bets · rolling the bankroll forward',23,bold=True)
    text(d,35,399,'Start 1,000 units · filtered legacy variance · forward predictions, 2020–21 to 2022–23',12)
    text(d,35,376,'All bets on a date use the opening balance. Reinvestment happens after that date settles.',11)
    dates=pd.to_datetime(curves.date)
    start=dates.min()-pd.Timedelta(days=1)
    low=min(1000,curves.balance.min())-40
    high=max(1000,curves.balance.max())+40
    x=lambda day:65+850*(day-start).days/(dates.max()-start).days
    y=lambda value:60+220*(value-low)/(high-low)
    for tick in sorted({round(low),1000,round(high)}):
        d.add(Line(65,y(tick),915,y(tick),strokeColor=HexColor('#d0d9d9'),strokeWidth=.7))
        text(d,57,y(tick)-3,str(tick),10,textAnchor='end')
    for i,(name,_,color) in enumerate(methods):
        values=curves[curves.method==name]
        text(d,35+(i%2)*470,347-(i//2)*24,f'{name} · final {values.balance.iat[-1]:,.2f}',12,color=color,bold=True)
        points=[x(start),y(1000)]
        for day,value in zip(pd.to_datetime(values.date),values.balance):
            points.extend([x(day),y(value)])
        d.add(PolyLine(points,strokeColor=HexColor(color),strokeWidth=1.5,fillColor=None))
    for date in [dates.min(),pd.Timestamp('2021-08-01'),pd.Timestamp('2022-08-01'),dates.max()]:
        text(d,x(date),36,date.strftime('%b %Y'),10,textAnchor='middle')
    with pdfium.PdfDocument(renderPDF.drawToString(d)) as doc:
        page=doc[0]
        bitmap=page.render(scale=1.4)
        bitmap.to_pil().save(OUT/'rolling_outcome_bankroll.png')
        bitmap.close()
        page.close()


if __name__=='__main__':
    main()
