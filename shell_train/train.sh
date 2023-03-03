ROOTDIR="/nas/k8s/dev/research/doyoungi/croptype_cls/utae-paps/"
WEIGHT_FOLDER=${ROOTDIR}/work_dir/test
TARGET_WEIGHT_FOLDER=${ROOTDIR}/work_dir/target_test
cd ${ROOTDIR}

# train for pastis
python3 ${ROOTDIR}/train_semantic.py --dataset_folder ${ROOTDIR}/datasets/mini_pastis_dataset \
    --res_dir ${WEIGHT_FOLDER} --model utae --rgbn --batch_size 1 --num_workers 4 --out_conv "[32, 20]" --num_classes 20 \
    --fold 1 --epoch 1 --lr 0.001

# train for custom dataset
python3 ${ROOTDIR}/train_semantic.py --dataset_folder ${ROOTDIR}/datasets/mini_target_dataset \
    --res_dir ${TARGET_WEIGHT_FOLDER} --model utae --rgbn --custom-dataset --batch_size 1 --num_workers 4 --out_conv "[32, 2]" --num_classes 2 \
    --weight_folder ${WEIGHT_FOLDER} --timestep 7 --fold 1 --epoch 1 --lr 0.0001

python3 ${ROOTDIR}/scene_test_semantic.py --weight-folder ${TARGET_WEIGHT_FOLDER}