declare -a ROS_VERSIONS=( "foxy" "galactic" "humble" "rolling" )

ORGANIZATION="microsoft"
MODEL_NAME="ssd"
declare -a MODEL_VERSIONS=( "<model versions here>" )

for VERSION in "${ROS_VERSIONS[@]}"
do
  ROS_VERSION="$VERSION"
  TAG="$MODEL_VERSION"
  gcloud builds submit --config cloudbuild.yaml . --substitutions=_ROS_VERSION="$ROS_VERSION" --timeout=10000 &
  pids+=($!)
  echo Dispatched "$MODEL_VERSION" on ROS "$ROS_VERSION"
done

for pid in ${pids[*]}; do
  wait "$pid"
done

echo "All builds finished"
