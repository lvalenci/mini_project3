########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 HMM helper
########################################

import re
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib import animation
from matplotlib.animation import FuncAnimation


####################
# WORDCLOUD FUNCTIONS
####################

def mask():
    # Parameters.
    r = 128
    d = 2 * r + 1

    # Get points in a circle.
    y, x = np.ogrid[-r:d-r, -r:d-r]
    circle = (x**2 + y**2 <= r**2)

    # Create mask.
    mask = 255 * np.ones((d, d), dtype=np.uint8)
    mask[circle] = 0

    return mask

def text_to_wordcloud(text, max_words=50, title='', show=True):
    plt.close('all')

    # Generate a wordcloud image.
    wordcloud = WordCloud(random_state=0,
                          max_words=max_words,
                          background_color='white',
                          mask=mask()).generate(text)

    # Show the image.
    if show:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=24)
        plt.show()

    return wordcloud

def states_to_wordclouds(hmm, obs_map, max_words=50, show=True):
    # Initialize.
    M = 100000
    n_states = len(hmm.A)
    obs_map_r = obs_map_reverser(obs_map)
    wordclouds = []

    # Generate a large emission.
    emission, states = hmm.generate_emission(M)

    # For each state, get a list of observations that have been emitted
    # from that state.
    obs_count = []
    for i in range(n_states):
        obs_lst = np.array(emission)[np.where(np.array(states) == i)[0]]
        obs_count.append(obs_lst)

    # For each state, convert it into a wordcloud.
    for i in range(n_states):
        obs_lst = obs_count[i]
        sentence = [obs_map_r[j] for j in obs_lst]
        sentence_str = ' '.join(sentence)

        wordclouds.append(text_to_wordcloud(sentence_str, max_words=max_words, title='State %d' % i, show=show))

    return wordclouds


####################
# HMM FUNCTIONS
####################

def parse_observations(text):
    # Convert text to dataset.
    lines = [line.split() for line in text.split('\n') if line.split()]

    obs_counter = 0
    obs = []
    obs_map = {}

    for line in lines:
        obs_elem = []
        
        for word in line:
            word = re.sub(r'[^\w]', '', word).lower()
            if word not in obs_map:
                # Add unique words to the observations map.
                obs_map[word] = obs_counter
                obs_counter += 1
            
            # Add the encoded word.
            obs_elem.append(obs_map[word])
        
        # Add the encoded sequence.
        obs.append(obs_elem)

    # Now, we reverse all observations before returning.
    for obs_index in range(len(obs)):
        obs[obs_index].reverse()

    return obs, obs_map

def obs_map_reverser(obs_map):
    obs_map_r = {}

    for key in obs_map:
        obs_map_r[obs_map[key]] = key

    return obs_map_r

def sample_sentence(hmm, obs_map, n_words=100):
    # Get reverse map.
    obs_map_r = obs_map_reverser(obs_map)

    # Sample and convert sentence.
    emission, states = hmm.generate_emission(n_words)
    sentence = [obs_map_r[i] for i in emission]

    return ' '.join(sentence).capitalize()

####################
# SONNET GENERATING FUNCTIONS
####################

def sample_sonnet(hmm, obs_map, n_words):
    # Generate 14 lines ("sentences"), i.e. generate 14 emissions
    sonnetLines = []
    sonnet = ''

    for numLines in range(14):
        line = sample_sentence(hmm, obs_map, n_words)
        sonnetLines.append(''.join(line).capitalize() + '\n')

    for line in sonnetLines:
        sonnet += line

    return sonnet 

def syllable_dict():
    """
    Returns dictionary of syllable counts as reported by Syllable_dictionary.txt
    <word>_ means syllable count of the word when it occurs at the end of a line.
    Keys of the dictionary are words and values of the dictionary are how many
    syllables that word has. 
    """
    counts = dict()
    
    with open('data/Syllable_dictionary.txt') as file:
        for line in file:
            while ',' in line:
                line = line.replace(',', '')
            while ':' in line:
                line = line.replace(':', '')
            while ';' in line:
                line = line.replace(';', '')
            while '.' in line:
                line = line.replace('.', '')
            while '(' in line:
                line = line.replace('(', '')
            while ')' in line:
                line = line.replace(')', '')
            while '?' in line:
                line = line.replace('?', '')
            while '!' in line:
                line = line.replace('!', '')
            arr = line.split(' ', 1)
            if 'E' in arr[1]:
                cts = arr[1].split(' ', 1)
                counts[arr[0]] = int(cts[1][0])
                counts[(arr[0] + "_")] = int(cts[0][1])
            else:
                counts[arr[0]] = int(arr[1][0])
    return counts

def sample_sentence_syl(hmm, obs_map, rhyme_dict, start_word, n_words=100):
    # Get reverse map.
    obs_map_r = obs_map_reverser(obs_map)

    num_start_word = obs_map[re.sub(r'[^\w]', '', start_word).lower()]
    num_rhyme_dict = {}

    # Convert the rhyme_dict to be composed of numbers instead of words.
    for _, (key, value) in enumerate(rhyme_dict.items()):
        num_value = []
        for val in value:
            # Clean up the word so we can see where it is in obs_map
            n_val = re.sub(r'[^\w]', '', val).lower()
            num_value.append(obs_map[n_val]) 

        n_key = re.sub(r'[^\w]', '', key).lower()
        num_rhyme_dict[obs_map[n_key]] = num_value

    # Sample and convert sentence.
    # emission, states = hmm.generate_emission(n_words, num_rhyme_dict)
    emission, states = hmm.generate_emission(n_words, num_start_word)
    sentence = [obs_map_r[i] for i in emission]

    # Flip the order of the sentence before returning.
    # sentence.reverse() 

    return sentence

def make_line(line, n_syl, syl_counts):
    """
    Given a line, makes a string consisting of first n_syl, of the line. 
    Returns tuple of whether line was successfully made and new line.
    Note: the lines fed into this function are REVERSED lines.
    """

    # Capitlize all i's to I's.
    for word_num in range(len(line)):
        if line[word_num] == 'i':
            line[word_num] = 'I'

    # Current number of syllables in constructed line.
    # This includes the syllable count of the first word.
    curr = 0

    # Now, since the list is reversed, the last word of the actual sonnet
    # line is the first word of 'line'. So we want to check if this
    # word can be counted as one syllable.

    # Number of syllable in first word (last word of actual line)
    init_syl = syl_counts[line[0]]
    init_syl_alt = init_syl

    # Alternative syllable count
    if ((line[0] + '_') in syl_counts):
        init_syl_alt = syl_counts[line[0] + '_']

    for i in range(1, n_syl):
        if line[i] not in syl_counts:
            return (False, '')

        w_syl = syl_counts[line[i]]

        if init_syl + curr + w_syl and init_syl_alt + curr + w_syl > n_syl:
            return (False, '')
        if init_syl+ curr + w_syl == n_syl or init_syl_alt + curr + w_syl == n_syl:
            return (True, ' '.join(line[:i+1]))
        curr += w_syl

def sample_sentence_syl_only(hmm, obs_map, n_words=100):
    # Get reverse map.
    obs_map_r = obs_map_reverser(obs_map)

    # Sample and convert sentence.
    emission, states = hmm.generate_emission(n_words)
    sentence = [obs_map_r[i] for i in emission]

    return sentence     

def make_line_syl_only(line, n_syl, syl_counts):
    """given a line, makes a string consisting of first n_syl, of the line. returns 
    tuple of whether line was successfully made and new line"""
    curr = 0
    
    # Capitlize all i's to I's.
    for word_num in range(len(line)):
        if line[word_num] == 'i':
            line[word_num] = 'I'
    
    for i in range(n_syl):
        if line[i] not in syl_counts:
            return (False, '')
        w_syl = syl_counts[line[i]]
        if curr + w_syl > n_syl:
            if ((line[i] + '_') in syl_counts) and \
                        (syl_counts[line[i] + '_'] + curr == n_syl):
                return (True, ' '.join(line[:i+1]).capitalize() + '\n')
            return (False, '')
        if curr + w_syl == n_syl:
            return (True, ' '.join(line[:i+1]).capitalize() + '\n')
        curr += w_syl

def sample_sonnet_syllables_only(hmm, obs_map, n_syl = 10):
    """samples a sonnect with n_syl number of syllables"""
    sonnetLines = []
    sonnet = ''
    sonnet_length = 14
    count = 0
    syl_counts = syllable_dict()
    
    while count < sonnet_length:
        line = sample_sentence_syl_only(hmm, obs_map, n_syl)
        (worked, nline) = make_line_syl_only(line, n_syl, syl_counts)
        if worked:
            sonnetLines.append(nline)
            count += 1
    for line in sonnetLines:
        sonnet += line
    return sonnet

def sample_sonnet_syl_and_rhyme(hmm, obs_map, rhyme_dict, n_syl = 10):
    """samples a sonnect with n_syl number of syllables and with 
    ababcdcdefefgg rhyme scheme"""
    sonnetLines = []
    r_sonnetLines = []
    sonnet = ''
    sonnet_length = 14
    count = 0
    syl_counts = syllable_dict()
    # print(syl_counts)
    
    while count < sonnet_length:
        # Pick a random word from the rhyming dictionary that the line has to start with. 
        
        start_word = np.random.choice(list(rhyme_dict.keys()))
        rhyme_word = np.random.choice(rhyme_dict[start_word])
        line1 = sample_sentence_syl(hmm, obs_map, rhyme_dict, start_word, n_syl)
        line2 = sample_sentence_syl(hmm, obs_map, rhyme_dict, rhyme_word, n_syl)
        (worked1, nline1) = make_line(line1, n_syl, syl_counts)
        (worked2, nline2) = make_line(line2, n_syl, syl_counts)
        if worked1 and worked2:
            sonnetLines.append(nline1)
            sonnetLines.append(nline2)
            count += 2

    # Now flip the order of each line.
    for line in sonnetLines:
        line_reversed = ' '.join(reversed(line.split(' '))).capitalize()
        r_sonnetLines.append(line_reversed + '\n')

    # Rearrange 7 couplets into a sonnet.
    for stanza in range(0, 3):
        idx = [0, 1, 2, 3]
        for i in range(len(idx)):
            idx[i] += stanza * 4
        sonnet += r_sonnetLines[idx[0]] + r_sonnetLines[idx[2]] + r_sonnetLines[idx[1]] + r_sonnetLines[idx[3]]

    for line_num in range(12, 14):
        sonnet += r_sonnetLines[line_num]

    return sonnet



######################
# MAKING SONNETS RHYME
######################

def create_rhyme_dict(text):
    '''
    This method does the same thing as parse_observations,
    but instead of creating an observation map for every word in
    Shakespeare's sonnets, it creates a dictionary where keys are
    words and values are words that rhyme with the keys.
    '''
    lines = [line.split() for line in text.split('\n') if line.split()]

    end_words = []
    sonnet_end_words = []

    rhyme_dict = {}

    # Sonnet 99
    rhyme_dict['chide'] = ['dyed']
    rhyme_dict['pride'] = ['dyed']
    rhyme_dict['dyed'] = ['chide', 'pride']

    for line_num in range(len(lines)):
        line = lines[line_num]
        # Sonnet 99 and 126 are strange.
        if line_num != 1376 and line_num not in range(1751, 1763):
            end_words.append(line[len(line) - 1])
        # Sonnet 126 has rhyme scheme aabb ccdd eeff.
        elif line_num in range(1751, 1763):
            sonnet_end_words.append(line[len(line) - 1])

    for line_num in range(len(sonnet_end_words)):
        word = sonnet_end_words[line_num]

        if (line_num % 2 == 0):
            if (line_num + 1) < len(sonnet_end_words):
                if word not in rhyme_dict.keys():
                    rhyme_dict[word] = [sonnet_end_words[line_num + 1]]
                elif sonnet_end_words[line_num + 1] not in rhyme_dict[word]:
                    rhyme_dict[word].append(sonnet_end_words[line_num + 1])
        elif (line_num % 2 == 1):
            if word not in rhyme_dict.keys():
                rhyme_dict[word] = [sonnet_end_words[line_num - 1]]
            elif sonnet_end_words[line_num - 1] not in rhyme_dict[word]:
                rhyme_dict[word].append(sonnet_end_words[line_num - 1])

    # Now, we use the fact that sonnets have an
    # abab cdcd efef gg rhyme scheme.
    # NOTE: Sonnet 99 is actually 15 lines,
    # with an ababa cdcd efef gg scheme.
    # Occurs at line 1376.

    # Only three of Shakespeare's 154 sonnets do not conform to this structure: 
    # Sonnet 99, which has 15 lines; 
    # Sonnet 126, which has 12 lines; 
    # and Sonnet 145, which is written in iambic tetrameter.

    for line_num in range(len(end_words)):
        word = end_words[line_num]

        if (line_num % 14 == 0) or (line_num % 14 == 1) or \
            (line_num % 14 == 4) or (line_num % 14 == 5) or \
            (line_num % 14 == 8) or (line_num % 14 == 9):
            if (line_num + 2) < len(end_words):
                # There may be multiple words which rhyme with a 
                # single word.
                if word not in rhyme_dict.keys():
                    rhyme_dict[word] = [end_words[line_num + 2]]
                elif end_words[line_num + 2] not in rhyme_dict[word]:
                    rhyme_dict[word].append(end_words[line_num + 2])
        elif (line_num % 14 == 2) or (line_num % 14 == 3) or \
            (line_num % 14 == 6) or (line_num % 14 == 7) or \
            (line_num % 14 == 10) or (line_num % 14 == 11):
            if word not in rhyme_dict.keys():
                rhyme_dict[word] = [end_words[line_num - 2]]
            elif end_words[line_num - 2] not in rhyme_dict[word]:
                rhyme_dict[word].append(end_words[line_num - 2])
        elif (line_num % 14 == 12):
            if (line_num + 1) < len(end_words):
                if word not in rhyme_dict.keys():
                    rhyme_dict[word] = [end_words[line_num + 1]]
                elif end_words[line_num + 1] not in rhyme_dict[word]:
                    rhyme_dict[word].append(end_words[line_num + 1])
        elif (line_num % 14 == 13):
            if word not in rhyme_dict.keys():
                rhyme_dict[word] = [end_words[line_num - 1]]
            elif end_words[line_num - 1] not in rhyme_dict[word]:
                rhyme_dict[word].append(end_words[line_num - 1])

    return rhyme_dict


####################
# HMM VISUALIZATION FUNCTIONS
####################

def visualize_sparsities(hmm, O_max_cols=50, O_vmax=0.1):
    plt.close('all')
    plt.set_cmap('viridis')

    # Visualize sparsity of A.
    plt.imshow(hmm.A, vmax=1.0)
    plt.colorbar()
    plt.title('Sparsity of A matrix')
    plt.show()

    # Visualize parsity of O.
    plt.imshow(np.array(hmm.O)[:, :O_max_cols], vmax=O_vmax, aspect='auto')
    plt.colorbar()
    plt.title('Sparsity of O matrix')
    plt.show()


####################
# HMM ANIMATION FUNCTIONS
####################

def animate_emission(hmm, obs_map, M=8, height=12, width=12, delay=1):
    # Parameters.
    lim = 1200
    text_x_offset = 40
    text_y_offset = 80
    x_offset = 580
    y_offset = 520
    R = 420
    r = 100
    arrow_size = 20
    arrow_p1 = 0.03
    arrow_p2 = 0.02
    arrow_p3 = 0.06
    
    # Initialize.
    n_states = len(hmm.A)
    obs_map_r = obs_map_reverser(obs_map)
    wordclouds = states_to_wordclouds(hmm, obs_map, max_words=20, show=False)

    # Initialize plot.    
    fig, ax = plt.subplots()
    fig.set_figheight(height)
    fig.set_figwidth(width)
    ax.grid('off')
    plt.axis('off')
    ax.set_xlim([0, lim])
    ax.set_ylim([0, lim])

    # Plot each wordcloud.
    for i, wordcloud in enumerate(wordclouds):
        x = x_offset + int(R * np.cos(np.pi * 2 * i / n_states))
        y = y_offset + int(R * np.sin(np.pi * 2 * i / n_states))
        ax.imshow(wordcloud.to_array(), extent=(x - r, x + r, y - r, y + r), aspect='auto', zorder=-1)

    # Initialize text.
    text = ax.text(text_x_offset, lim - text_y_offset, '', fontsize=24)
        
    # Make the arrows.
    zorder_mult = n_states ** 2 * 100
    arrows = []
    for i in range(n_states):
        row = []
        for j in range(n_states):
            # Arrow coordinates.
            x_i = x_offset + R * np.cos(np.pi * 2 * i / n_states)
            y_i = y_offset + R * np.sin(np.pi * 2 * i / n_states)
            x_j = x_offset + R * np.cos(np.pi * 2 * j / n_states)
            y_j = y_offset + R * np.sin(np.pi * 2 * j / n_states)
            
            dx = x_j - x_i
            dy = y_j - y_i
            d = np.sqrt(dx**2 + dy**2)

            if i != j:
                arrow = ax.arrow(x_i + (r/d + arrow_p1) * dx + arrow_p2 * dy,
                                 y_i + (r/d + arrow_p1) * dy + arrow_p2 * dx,
                                 (1 - 2 * r/d - arrow_p3) * dx,
                                 (1 - 2 * r/d - arrow_p3) * dy,
                                 color=(1 - hmm.A[i][j], ) * 3,
                                 head_width=arrow_size, head_length=arrow_size,
                                 zorder=int(hmm.A[i][j] * zorder_mult))
            else:
                arrow = ax.arrow(x_i, y_i, 0, 0,
                                 color=(1 - hmm.A[i][j], ) * 3,
                                 head_width=arrow_size, head_length=arrow_size,
                                 zorder=int(hmm.A[i][j] * zorder_mult))

            row.append(arrow)
        arrows.append(row)

    emission, states = hmm.generate_emission(M)

    def animate(i):
        if i >= delay:
            i -= delay

            if i == 0:
                arrows[states[0]][states[0]].set_color('red')
            elif i == 1:
                arrows[states[0]][states[0]].set_color((1 - hmm.A[states[0]][states[0]], ) * 3)
                arrows[states[i - 1]][states[i]].set_color('red')
            else:
                arrows[states[i - 2]][states[i - 1]].set_color((1 - hmm.A[states[i - 2]][states[i - 1]], ) * 3)
                arrows[states[i - 1]][states[i]].set_color('red')

            # Set text.
            text.set_text(' '.join([obs_map_r[e] for e in emission][:i+1]).capitalize())

            return arrows + [text]

    # Animate!
    print('\nAnimating...')
    anim = FuncAnimation(fig, animate, frames=M+delay, interval=1000)

    return anim

    # honestly this function is so jank but who even fuckin cares
    # i don't even remember how or why i wrote this mess
    # no one's gonna read this
    # hey if you see this tho hmu on fb let's be friends
