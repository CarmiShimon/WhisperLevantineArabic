import glob
import re


# swap_tags = {
#     '{HESITATION}': 'آه',
#     '{STAMMER}': 'أأأ',
#     '{NOD}': 'ممم'

regex_letters = r'[a-zA-Z]'
valid_tags = ['{MUSIC}', '{SPEAKER CHANGE}', '{PHONE}', '{HUMAN NOISE}', '{MOUTH NOISE}', '{LAUGH}',
              '{BREATH}', '{CAUGH}', '{UNHUMAN NOISE}', '{VEHICLE}', '{PARALLEL}', '{TELEVISION}', '{MUSIC}',
              '{TYPING}', '{DOOR}',	'{CRYING}']
swap_tags = {'{HESITATION}': 'آه', '{STAMMER}': 'أأأ', '{NOD}': 'ممم'}
invalid_tags = ['{UNK}', '{DISTORTION}']
english_words = {
    'tech': 'تيك',
    'dot': 'دوت',
    'text': 'تكست',
    'one': 'ون',
    'business': 'بزنس',
    'phone': 'فون',
    'label': 'ليبل',
    'access': 'اكسس',
    'galaxy': 'جالاكسي',
    'gym': 'جيم',
    'apple': 'أبل'
}
# max_num_tags = 4

def filter_text(text):
    for v in valid_tags:
        if v in text:
            text = text.replace(v, '')
    for key, value in swap_tags.items():
        if key in text:
            text = text.replace(key, value)
    for inv in invalid_tags:
        if inv in text:
            return ''
    for key, value in english_words.items():
        if key in text:
            text = text.replace(key, value)
    return text

def convert_decimal_seconds(decimal_seconds):
    from datetime import timedelta

    # Convert decimal seconds to total milliseconds
    total_milliseconds = int(decimal_seconds * 1000)

    # Extract hours, minutes, seconds, and milliseconds
    hours = total_milliseconds // (3600 * 1000)
    minutes = (total_milliseconds % (3600 * 1000)) // (60 * 1000)
    seconds = (total_milliseconds % (60 * 1000)) // 1000
    milliseconds = total_milliseconds % 1000

    # Format as HH:MM:SS.mmm with exactly 3-digit milliseconds
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

def stm_to_vtt(stm_file):
    print(stm_file)
    with open(stm_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    vtt_file = stm_file[:-4] + '.vtt'
    with open(vtt_file, 'w', encoding='utf-8') as vtt:
        vtt.write("WEBVTT\n\n")

        for i, line in enumerate(lines):
            # split milestone_1_and_2
            if 'milestone_1_and_2' in stm_file:
                parts = line.strip().split(' ', 6)  # Splitting into 7 parts
                start_time = float(parts[3])
                end_time = float(parts[4])
                text = parts[6]
                # Convert timestamps to VTT format (hh:mm:ss.sss)
                start_vtt = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{start_time % 60:06.3f}"
                end_vtt = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{end_time % 60:06.3f}"
            elif ('milestone_3' in stm_file) or ('milestone_4_1' in stm_file) or ('milestone_4_2' in stm_file) \
                    or ('milestone_5' in stm_file) or ('milestone_6' in stm_file):
                parts = line.strip().split(' ', 5)  # Splitting into 6 parts
                start_time = parts[3]
                end_time = parts[4]
                text = parts[5][4:]
                # Convert timestamps to VTT format (hh:mm:ss.sss)
                start_vtt = convert_decimal_seconds(float(start_time))
                end_vtt = convert_decimal_seconds(float(end_time))

            # filter line by text:
            text = filter_text(text)
            if (text == '') or re.search(regex_letters, text):
                text = '-100'

            vtt.write(f"{start_vtt} --> {end_vtt}\n{text}\n\n")

    print(f"Converted {stm_file} to {vtt_file}")


if __name__ == '__main__':
    data_dir = r'/shares/rndsounds/wake_up_word/genAI/ac/stt/'
    for sub_dir in ['milestone_1_and_2', 'milestone_3', 'milestone_4_1', 'milestone_4_2', 'milestone_5', 'milestone_6']:
    # for sub_dir in ['milestone_5']:
        data_dir_path = data_dir + sub_dir
        for stm_file in glob.glob(data_dir_path + '/stm_clean/*.stm'):
            stm_to_vtt(stm_file)
