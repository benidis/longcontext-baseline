# Group definitions shared by run_group.sh and resume_group.sh.
# Source this file, then use: resolve_group <group_id>
# Sets GPUS and CONFIGS arrays.

resolve_group() {
    local GROUP="$1"
    case "${GROUP}" in
        ec2_1_gpu0)
            GPUS="0,1,2,3"
            CONFIGS=(
                "128k/clinc150"
            )
            ;;
        ec2_1_gpu4)
            GPUS="4,5,6,7"
            CONFIGS=(
                "128k/nlu"
                "128k/infinite_bench_mc"
            )
            ;;
        ec2_2_gpu0)
            GPUS="0,1,2,3"
            CONFIGS=(
                "128k/trec_coarse"
                "64k/hotpot_qa"
                "128k/json_kv"
                "64k/pop_qa"
            )
            ;;
        ec2_2_gpu4)
            GPUS="4,5,6,7"
            CONFIGS=(
                "128k/nq"
                "64k/nq"
                "64k/trivia_qa"
                "128k/pop_qa"
            )
            ;;
        ec2_3_gpu0)
            GPUS="0,1,2,3"
            CONFIGS=(
                "128k/hotpot_qa"
                "128k/trivia_qa"
                "128k/infinite_bench_qa"
            )
            ;;
        ec2_3_gpu4)
            GPUS="4,5,6,7"
            CONFIGS=(
                "128k/banking77"
                "128k/trec_fine"
            )
            ;;
        *)
            echo "Unknown group: ${GROUP}"
            echo "Valid group IDs: ec2_1_gpu0 ec2_1_gpu4 ec2_2_gpu0 ec2_2_gpu4 ec2_3_gpu0 ec2_3_gpu4"
            return 1
            ;;
    esac
}
