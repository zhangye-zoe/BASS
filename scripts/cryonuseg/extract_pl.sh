
GPUS=$1
RESULTNAME=$2
ANNFILE=$3

bash tools/dist_test.sh configs/noisyboundaries/cryonuseg/mask_rcnn_r50_fpn_1x_coco_generatepl.py work_dirs/CryoNuSeg-1-2/mask_rcnn_r50_fpn_1x_coco_sup/epoch_15.pth $GPUS --out $RESULTNAME

python scripts/cryonuseg/pkl2json.py $RESULTNAME

python scripts/cryonuseg/filter_pl.py $RESULTNAME

python scripts/cryonuseg/form_ann.py $RESULTNAME $ANNFILE
