from utils.vtt import vtt_to_whisper_result
import glob
import json


if __name__ == '__main__':
    data_dir = r'/shares/rndsounds/wake_up_word/genAI/ac/stt/'
    total_duration = 0
    duration_per_milestone = 0
    for sub_dir in ['milestone_1_and_2', 'milestone_3', 'milestone_4_1', 'milestone_4_2', 'milestone_5', 'milestone_6']:
    # for sub_dir in ['milestone_5']:
        data_dir_path = data_dir + sub_dir
        for vtt_file in glob.glob(data_dir_path + '/stm_clean//*.vtt'):
            whisper_json = vtt_to_whisper_result(vtt_file)
            whisper_json.save_as_json(f'{vtt_file[:-4]}.json')
            total_duration += sum([segment.end - segment.start for segment in whisper_json.segments])
            # print(f'Done converting {vtt_file}')
            duration_per_milestone += sum([segment.end - segment.start for segment in whisper_json.segments])
        print(f'Duration per milestone {sub_dir} = {duration_per_milestone/3600:.2f} hours')
        duration_per_milestone = 0  # Reset for the next milestone
    print(f'Total duration of all segments: {total_duration/3600:.2f} hours')