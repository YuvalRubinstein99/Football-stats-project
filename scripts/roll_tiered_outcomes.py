"""Compound the existing outcome stake tiers using each day's opening bankroll."""
from roll_outcome_bankroll import (
    forward_goal_forecasts, OUT, pd, np, outcome_probabilities, selections, SEARCH_CANDIDATE,
    Drawing, Rect, Line, PolyLine, HexColor, text, renderPDF, pdfium,
)


def main():
    parts=[]
    for _,_,frame in forward_goal_forecasts():
        p=outcome_probabilities(frame[['mean_home','mean_away']].to_numpy())
        picks=selections(frame,p)[SEARCH_CANDIDATE]
        part=frame[['date','season','home_team_name','away_team_name']].copy()
        part=part.assign(bet=picks[:,0],tier=picks[:,1],unit_net=picks[:,4])
        parts.append(part[part.bet==1])
    bets=pd.concat(parts).sort_values('date')
    previous=pd.read_csv(OUT/'rolling_outcome_bets.csv').sort_values(
        ['date','home_team_name','away_team_name'])
    ordered=bets.sort_values(['date','home_team_name','away_team_name'])
    assert ordered[['home_team_name','away_team_name']].values.tolist()==previous[['home_team_name','away_team_name']].values.tolist()
    np.testing.assert_allclose(ordered.unit_net,previous.unit_net)
    assert bets.tier.between(1,5).all()
    rows,paths=[],[]
    rates=[0,.001,.005,.01]
    for rate in rates:
        for method in ['Flat','Tiered']:
            balance=peak=1000.
            drawdown=spent=0.
            capped_days=0
            for date,group in bets.groupby('date'):
                base=balance*rate if rate else 1.
                stakes=base*(group.tier.to_numpy() if method=='Tiered' else np.ones(len(group)))
                total=stakes.sum()
                if total>balance:
                    stakes*=balance/total
                    capped_days+=1
                spent+=stakes.sum()
                balance+=float(stakes@group.unit_net.to_numpy())
                peak=max(peak,balance)
                drawdown=max(drawdown,1-balance/peak)
                assert balance>=-1e-8
                paths.append(dict(rate=rate,method=method,date=date,balance=balance))
            rows.append(dict(rate=rate,method=method,bets=len(bets),final=balance,net=balance-1000,
                             roi=(balance-1000)/spent,staked=spent,drawdown=drawdown,capped_days=capped_days))
    summary=pd.DataFrame(rows)
    curves=pd.DataFrame(paths)
    old=pd.read_csv(OUT/'rolling_outcome_summary.csv')
    np.testing.assert_allclose(summary[summary.method=='Flat']['final'],old.ending_balance)
    summary.to_csv(OUT/'rolling_tiers_summary.csv',index=False)
    curves.to_csv(OUT/'rolling_tiers_curves.csv',index=False)
    bets.to_csv(OUT/'rolling_tiers_bets.csv',index=False)
    print(summary.to_string(index=False))
    d=Drawing(1160,1120)
    d.add(Rect(0,0,1160,1120,fillColor=HexColor('#ffffff'),strokeColor=None))
    text(d,32,1080,'Outcome bankroll · flat stakes versus tiers',25,bold=True)
    text(d,32,1054,'1,000-unit start · 1,185 identical selections · 2020–21 through 2022–23',13)
    text(d,32,1032,'Daily reinvestment. Tiered stakes multiply the base stake by 1–5. Each row shares a vertical scale.',12)
    for row,rate in enumerate(rates):
        panel=curves[curves.rate==rate]
        low=min(1000,panel.balance.min())*.95
        high=max(1000,panel.balance.max())*1.05
        for col,(method,color) in enumerate([('Flat','#18766b'),('Tiered','#7953a4')]):
            left,right=65+col*575,535+col*575
            top=918-row*245
            bottom=top-142
            values=panel[panel.method==method]
            entry=summary[(summary.rate==rate)&(summary.method==method)].iloc[0]
            dates=pd.to_datetime(values.date)
            start=dates.min()-pd.Timedelta(days=1)
            x=lambda day:left+(right-left)*(day-start).days/(dates.max()-start).days
            y=lambda value:bottom+(top-bottom)*(value-low)/(high-low)
            base='Fixed 1-unit base' if rate==0 else f'{rate:.1%} bankroll base'
            text(d,left-25,top+65,base+' · '+method,16,bold=True,color=color)
            text(d,left-25,top+42,f'Final {entry.final:,.2f}   |   growth {(entry.final/1000-1):+.1%}',12,color=color)
            text(d,left-25,top+22,f'Max daily drawdown {entry.drawdown:.1%}   |   betting ROI {entry.roi:+.2%}',11)
            for tick in sorted({round(low),1000,round(high)}):
                d.add(Line(left,y(tick),right,y(tick),strokeColor=HexColor('#d0d9d9'),strokeWidth=.7))
                text(d,left-7,y(tick)-3,f'{tick:,}',9,textAnchor='end')
            points=[x(start),y(1000)]
            for day,value in zip(dates,values.balance):
                points.extend([x(day),y(value)])
            d.add(PolyLine(points,strokeColor=HexColor(color),strokeWidth=1.5,fillColor=None))
            for date in [dates.min(),pd.Timestamp('2021-08-01'),pd.Timestamp('2022-08-01'),dates.max()]:
                text(d,x(date),bottom-20,date.strftime('%b %Y'),10,textAnchor='middle')
    with pdfium.PdfDocument(renderPDF.drawToString(d)) as doc:
        page=doc[0]
        bitmap=page.render(scale=1.4)
        bitmap.to_pil().save(OUT/'rolling_tiers.png')
        bitmap.close()
        page.close()


if __name__=='__main__':
    main()
