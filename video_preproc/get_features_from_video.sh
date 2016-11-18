vidid=0
for f in "$1"/*mp4; do
  #echo ${f}
  #remove path
  video_name=${f##*/}
  video_name=${video_name%.mp4}
  #echo $video_name
  avconv -i "${f}" -r 1 -f image2 "$2"/"vid${vidid}"_f%d.jpg
  echo "${vidid} ${video_name}" >> mapping_youcook.txt
  vidid=$((vidid+1))
done
