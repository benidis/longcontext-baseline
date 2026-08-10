# Group definitions shared by run_group.sh and resume_group.sh.
# Source this file, then use: resolve_group <group_id>
# Sets GPUS and CONFIGS arrays.

resolve_group() {
    local GROUP="$1"
    case "${GROUP}" in
        ec2_1_gpu0)
            GPUS="0,1,2,3"
            CONFIGS=(
                "128k/nlu"      
            )
            ;;
        ec2_1_gpu4)
            GPUS="4,5,6,7"
            CONFIGS=(
                "128k/clinc150"
            )
            ;;
        ec2_2_gpu0)
            GPUS="0,1,2,3"
            CONFIGS=(
                "128k/nq"
                "64k/nq"
                "64k/hotpot_qa"
                "128k/ms_macro"
                "64k/trivia_qa"
            )
            ;;
        ec2_2_gpu4)
            GPUS="4,5,6,7"
            CONFIGS=(
                "128k/trivia_qa"
                "128k/pop_qa"
                "128k/infinite_bench_qa"
                "128k/json_kv"
                "128k/ruler_mk_uuid"
            )
            ;;
        ec2_3_gpu0)
            GPUS="0,1,2,3"
            CONFIGS=(
                "128k/trec_coarse"
                "64k/nlu"
                "64k/trec_coarse"
                "64k/pop_qa"
                "64k/infinite_bench_mc"
                "64k/ms_macro"
            )
            ;;
        ec2_3_gpu4)
            GPUS="4,5,6,7"
            CONFIGS=(
                "128k/hotpot_qa"
                "64k/clinc150"
                "128k/infinite_bench_mc"
                "64k/infinite_bench_qa"
                "64k/ruler_mk_uuid"
                "64k/json_kv"
            )
            ;;
        *)
            echo "Unknown group: ${GROUP}"
            echo "Valid group IDs: ec2_1_gpu0 ec2_1_gpu4 ec2_2_gpu0 ec2_2_gpu4 ec2_3_gpu0 ec2_3_gpu4"
            return 1
            ;;
    esac
}
