from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import gaussian_kde
from PIL import Image
import shutil,json
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--input-dir',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
src=args.input_dir
p=args.output;p.mkdir(parents=True,exist_ok=True)
a=np.load(src/'verified_shap_arrays.npz');sv=a['shap_values'];z=a['transformed_features'];names=a['feature_names'].tolist()
t=pd.read_csv(src/'feature_label_mapping.csv').head(14)
plt.rcParams.update({'font.family':'Arial','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
fig,ax=plt.subplots(figsize=(12.8,8.2));fig.subplots_adjust(left=.305,right=.91,top=.88,bottom=.15)
cm=LinearSegmentedColormap.from_list('value',['#1683C4','#8064B3','#D93677'])
rng=np.random.default_rng(42);grid=np.linspace(-1.55,1.55,1000)
for j,row in t.iterrows():
 i=names.index(row.original_feature);v=sv[:,i];y=13-j
 # Scott bandwidth gives an explicitly smoothed descriptive density; raw x coordinates are unchanged.
 kde=gaussian_kde(v,bw_method='scott');d=kde(grid);d=d/d.max()*.33
 ax.fill_between(grid,y-d,y+d,facecolor='#C9DEE6',edgecolor='#89ABB9',linewidth=.65,zorder=2)
 f=z[:,i];lo,hi=np.percentile(f,[5,95]);c=np.clip((f-lo)/(hi-lo),0,1)
 ax.scatter(v,y+rng.uniform(-.13,.13,len(v)),c=c,cmap=cm,vmin=0,vmax=1,s=9,alpha=.6,linewidths=0,zorder=3)
ax.set_yticks(range(14),t.display_label.tolist()[::-1]);ax.tick_params(axis='y',length=0,pad=10)
ax.set_xlim(-1.55,1.55);ax.set_ylim(-.65,13.65);ax.axvline(0,color='#9EADB4',linewidth=.8,zorder=1)
ax.grid(axis='x',color='#E8EDF0',linewidth=.6);ax.set_axisbelow(True)
ax.set_xlabel('SHAP value (change in predicted IPSS)',labelpad=12)
ax.spines['bottom'].set_color('#9EADB4')
fig.text(.305,.957,'SHAP distribution with smoothed density',fontsize=16,weight='bold',color='#243746')
fig.text(.305,.925,'Top 14 features | Gradient boosting | Three-month IPSS | n = 148',color='#687C89')
cax=fig.add_axes([.936,.25,.012,.48]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap=cm),cax=cax)
cb.set_ticks([0,1],labels=['Low','High']);cb.outline.set_visible(False);cb.ax.tick_params(length=0);cb.set_label('Feature value')
fig.text(.305,.068,'Dots: observed SHAP values; vertical jitter only.',fontsize=9,color='#526575')
fig.text(.305,.044,'Shading: Gaussian KDE (Scott bandwidth); smoothing can span gaps without observations.',fontsize=9,color='#526575')
fig.text(.305,.020,'Density widths are normalized within each feature. Full-cohort descriptive explanation.',fontsize=9,color='#526575')
out=p/'smoothed';out.mkdir(exist_ok=True)
for ext in ['png','pdf','svg']:fig.savefig(out/('gradient_boosting_shap_smoothed_density.'+ext),dpi=350,facecolor='white')
plt.close(fig)
im=Image.open(out/'gradient_boosting_shap_smoothed_density.png').convert('RGB');im.thumbnail((1250,900));im.save(p/'smoothed_preview.jpg',quality=85)
(out/'caption.txt').write_text('Distribution of full-cohort SHAP values for the top 14 features of the gradient boosting model predicting three-month IPSS. Colored points indicate the original SHAP values for 148 patients; vertical jitter is used solely to reduce overlap. Shaded envelopes are Gaussian kernel density estimates using Scott bandwidth, normalized separately for each feature. Smoothing may extend across unobserved intervals and must not be interpreted as additional patient observations. Positive SHAP values increase predicted IPSS. These explanations are descriptive, not external validation.\n',encoding='utf-8')
print('RENDERED',flush=True)
