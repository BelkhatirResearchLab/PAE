from pathlib import Path
import json, re, shutil, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import shap
from PIL import Image
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--arrays',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
P=args.arrays.parent
O=args.output;O.mkdir(parents=True,exist_ok=True)
data=np.load(args.arrays)
sv=data['shap_values'];z=data['transformed_features'];names=data['feature_names'].tolist()
mean=np.abs(sv).mean(axis=0)
clinical=['age_at_pae','prostate_volume_preop_cc','abrams_griffiths_number','qmax_preop','pvr_preop','ipss_preop']
ref=pd.DataFrame(index=names)
labels={'shapeS_volume':'Shape | Volume','shapeS_filledVolume':'Shape | Filled volume','shapeS_flatness':'Shape | Flatness','ipss_preop':'Clinical | Baseline IPSS','abrams_griffiths_number':'Clinical | Abrams-Griffiths index','Original_szmFeatS_largeAreaEmphasis':'GLSZM | Large-area emphasis','Original_szmFeatS_largeAreaLowGrayLevelEmphasis':'GLSZM | Large-area low-GL emphasis','Original_szmFeatS_largeAreaHighGrayLevelEmphasis':'GLSZM | Large-area high-GL emphasis','Original_szmFeatS_sizeZoneVariance':'GLSZM | Zone-size variance','Original_glcmFeatS_AvgS_secondInfCorr':'GLCM | Information correlation 2','Original_ngldmFeatS_LowDependenceLowGrayLevelEmphasis':'NGLDM | Low-dependence low-GL','Original_firstOrderS_kurtosis':'First order | Kurtosis','Original_ngldmFeatS_Entropy':'NGLDM | Entropy','Original_ngldmFeatS_LowDependenceHighGrayLevelEmphasis':'NGLDM | Low-dependence high-GL','Original_szmFeatS_zonePercentage':'GLSZM | Zone percentage','Original_szmFeatS_sizeZoneNonUniformityNorm':'GLSZM | Normalized zone-size NU','Original_szmFeatS_smallAreaEmphasis':'GLSZM | Small-area emphasis','Original_ngldmFeatS_Energy':'NGLDM | Energy','Original_szmFeatS_sizeZoneNonUniformity':'GLSZM | Zone-size non-uniformity','Original_ngldmFeatS_DependenceCountVariance':'NGLDM | Dependence-count variance'}
order=sorted(range(len(names)),key=lambda i:(-mean[i],list(ref.index).index(names[i])))
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.labelsize':11,'axes.edgecolor':'#9BA8B1','text.color':'#243746','axes.labelcolor':'#243746','xtick.color':'#526575','ytick.color':'#243746','pdf.fonttype':42,'ps.fonttype':42})
cmap=LinearSegmentedColormap.from_list('feature_value',['#1683C4','#8064B3','#D93677'])
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(O/(name+'.'+ext),dpi=350,facecolor='white')
 plt.close(fig)
def plots(n,suffix=''):
 ids=order[:n]; labs=[labels.get(names[i],names[i]) for i in ids]
 height=7.6 if n==14 else 9.2
 fig,ax=plt.subplots(figsize=(12.8,height));fig.subplots_adjust(left=.305,right=.95,bottom=.115,top=.91)
 yy=np.arange(n); vals=mean[ids]
 colors=['#D78A35' if names[i] in clinical else '#26869B' for i in ids]
 ax.barh(yy,vals,height=.64,color=colors,zorder=3)
 ax.set_yticks(yy,labs,fontsize=10);ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.14)
 ax.set_xlabel('Mean absolute SHAP value (IPSS points)',labelpad=12)
 ax.grid(axis='x',color='#E5EBEF',linewidth=.7,zorder=0);ax.tick_params(axis='y',length=0,pad=10)
 for sp in ['top','right','left']:ax.spines[sp].set_visible(False)
 for j,v in enumerate(vals):ax.text(v+.009,j,f'{v:.3f}',va='center',fontsize=9,color='#526575')
 fig.text(.305,.955,'Global feature importance',fontsize=16,weight='bold')
 fig.text(.305,.922,f'Top {n} features | Gradient boosting | Three-month IPSS',fontsize=10,color='#687C89')
 fig.text(.305,.025,'Radiomics',color='#26869B',fontsize=10);fig.text(.405,.025,'Clinical',color='#D78A35',fontsize=10)
 save(fig,'gradient_boosting_shap_bar'+suffix)
 np.random.seed(42)
 shap.summary_plot(sv[:,ids],z[:,ids],feature_names=labs,max_display=n,sort=False,show=False,plot_size=(12.8,height),cmap=cmap,alpha=.75,color_bar=False)
 fig=plt.gcf();ax=plt.gca();fig.subplots_adjust(left=.305,right=.91,bottom=.115,top=.91)
 ax.tick_params(axis='y',labelsize=10,pad=10);ax.tick_params(axis='x',labelsize=10)
 ax.set_xlabel('SHAP value (change in predicted IPSS)',fontsize=11,labelpad=12)
 ax.set_xlim(float(sv[:,ids].min())-.16,float(sv[:,ids].max())+.16)
 for col in ax.collections:col.set_sizes([15])
 fig.text(.305,.955,'How each feature shifts the prediction',fontsize=16,weight='bold')
 fig.text(.305,.922,f'Top {n} features | Each dot represents one patient (n = 148)',fontsize=10,color='#687C89')
 cax=fig.add_axes([.936,.22,.012,.55]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap=cmap),cax=cax)
 cb.set_ticks([0,1],labels=['Low','High']);cb.outline.set_visible(False);cb.ax.tick_params(length=0,labelsize=9);cb.set_label('Feature value',fontsize=10,labelpad=8)
 fig.text(.305,.025,'Lower predicted IPSS',fontsize=10,color='#687C89');fig.text(.91,.025,'Higher predicted IPSS',ha='right',fontsize=10,color='#687C89')
 save(fig,'gradient_boosting_shap_beeswarm'+suffix)
plots(14);plots(20,'_top20')
print('Figures regenerated:',O)
