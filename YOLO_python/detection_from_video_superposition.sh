m="15"
l="carrot"

# carrots group A
for i in `seq 1 45`
do
    echo video: carrot_a$i
    python darknet_find_object_from_video.py -v ../../../dataset/carrots/videos/group_a/carrot_a$i.avi   -l carrot -od ../../../dataset/carrots/detected_by_gcnn/from_videos_superposition/group_a/carrot_a$i -m $m -superposition
    sleep 5
done

# carrots group B
for i in `seq 1 45`
do
    echo video: carrot_b$i
    python darknet_find_object_from_video.py -v ../../../dataset/carrots/videos/group_b/carrot_b$i.avi   -l carrot -od ../../../dataset/carrots/detected_by_gcnn/from_videos_superposition/group_b/carrot_b$i -m $m -superposition
    sleep 5
done

