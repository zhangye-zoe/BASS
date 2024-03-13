
GPUS=$1
RESULTNAME=$2
ANNFILE=$3

bash tools/dist_test.sh configs/noisyboundaries/kumar/mask_rcnn_r50_fpn_1x_coco_generatepl.py work_dirs/mask_rcnn_r50_fpn_1x_coco_sup/epoch_30.pth $GPUS --out $RESULTNAME

python scripts/kumar/pkl2json.py $RESULTNAME

python scripts/kumar/filter_pl.py $RESULTNAME

python scripts/kumar/form_ann.py $RESULTNAME $ANNFILE
