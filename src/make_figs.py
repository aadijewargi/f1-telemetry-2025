import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUTD=sys.argv[1] if len(sys.argv)>1 else "analysis_out"
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb","#0b0b0b","#52514e","#898781","#e1e0d9","#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6","#eb6834","#1baf7a"
T=OUTD+"/tables/"; F=OUTD+"/figures/"
R=json.load(open(OUTD+"/results.json"))
plt.rcParams.update({"figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,"axes.edgecolor":AXIS,
 "axes.labelcolor":INK2,"xtick.color":MUTED,"ytick.color":MUTED,"text.color":INK,"axes.grid":True,"grid.color":GRID,
 "grid.linewidth":1,"axes.spines.top":False,"axes.spines.right":False,"font.size":10,"axes.titleweight":"bold","axes.titlesize":12,
 "axes.titlelocation":"left","axes.axisbelow":True,"font.family":"DejaVu Sans"})
def save(fig,n): fig.tight_layout(); fig.savefig(F+n,dpi=170); plt.close(fig)

# fig3 braking energy
e=pd.read_csv(T+"circuit_energy.csv").sort_values("Ebrake_MJ")
fig,ax=plt.subplots(figsize=(8,6.2))
ax.barh(e.Event.str.replace(" Grand Prix",""),e.Ebrake_MJ,color=BLUE,height=.62)
ax.axvline(2,color=ORANGE,lw=2); ax.annotate("MGU-K recovery cap: 2 MJ per lap",xy=(2,1.5),xytext=(10.5,1.5),color=INK2,fontsize=9,va="center",arrowprops=dict(arrowstyle="-",color=MUTED))
ax.set_xlabel("Kinetic energy shed under braking per lap (MJ)"); ax.grid(axis="y",visible=False)
ax.set_title("Braking sheds ~5x more energy than the MGU-K may recover")
for y,v in enumerate(e.Ebrake_MJ): ax.text(v+.15,y,f"{v:.1f}",va="center",fontsize=8,color=INK2)
ax.set_xlim(0,16.5); save(fig,"fig3_braking_energy.png")

# fig4 full-throttle
c=pd.read_csv(T+"fullthrottle_curve.csv")
fig,(a,b)=plt.subplots(1,2,figsize=(9.5,4.2))
a.plot(c.SpeedBin_kph+5,c.a_g,color=BLUE,lw=2); a.set_xlabel("Speed (km/h)"); a.set_ylabel("Acceleration at full throttle (g)"); a.set_title("Acceleration fades with speed")
b.plot(c.SpeedBin_kph+5,c.Pkin_kW,color=BLUE,lw=2); b.set_xlabel("Speed (km/h)"); b.set_ylabel("m·a·v (kW)"); b.set_title("Power into kinetic energy")
b.set_ylim(0,None); save(fig,"fig4_fullthrottle.png")

# fig5 DRS
d=pd.read_csv(T+"drs_effect_by_round.csv")
fig,ax=plt.subplots(figsize=(7.2,4.6))
ax.scatter(d.top_kph,d.dCdA_m2,s=64,color=BLUE,edgecolor=SURF,linewidth=2,zorder=3)
k=np.polyfit(d.top_kph,d.dCdA_m2,1); xs=np.linspace(d.top_kph.min(),d.top_kph.max(),10); ax.plot(xs,np.polyval(k,xs),color=MUTED,lw=2)
for _,r in d[d.Event.str.contains("Las Vegas|Austrian|Italian|Monaco")].iterrows():
    ax.annotate(r.Event.replace(" Grand Prix",""),(r.top_kph,r.dCdA_m2),xytext=(6,6),textcoords="offset points",fontsize=8.5,color=INK2)
ax.set_xlabel("Median lap top speed (km/h)"); ax.set_ylabel("Drag-area reduction with DRS (m²)")
ax.set_title("DRS removes about 0.37 m² of drag area, less on fast tracks"); save(fig,"fig5_drs.png")

# fig6 lateral envelope
l=pd.read_csv(T+"lateral_envelope_pooled.csv"); P=R["lateral_pooled"]
fig,ax=plt.subplots(figsize=(7.2,4.6))
v=(l.bin+10)/3.6
ax.scatter(l.bin+10,l.p97,s=64,color=BLUE,edgecolor=SURF,linewidth=2,zorder=3)
vv=np.linspace(70,270,50)/3.6; beta=(P["grip_at_250_g"]-P["mu_g"])/(250/3.6)**2
ax.plot(vv*3.6,P["mu_g"]+beta*vv**2,color=ORANGE,lw=2)
ax.set_xlabel("Speed (km/h)"); ax.set_ylabel("97th-percentile lateral acceleration (g)")
ax.set_title("Cornering grip grows with speed: downforce at work")
ax.text(90,5.2,f"fit: {P['mu_g']:.2f} g + aero term\nR² = {P['r2']:.2f}",fontsize=9,color=INK2); save(fig,"fig6_lateral.png")
print("ok")
