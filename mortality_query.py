import numpy as np
import operator
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import math
from tkinter import *


# Functions age_encode, race_encode, state_encode, and self_core_dict used to create the core dict.
def age_encode(age):
    # Returns the utf-8 object of an arbitrary integer age input.
    if age < 1:
        return '1'.encode('utf-8')
    elif age < 5:
        return '1-4'.encode('utf-8')
    elif age < 10:
        return '5-9'.encode('utf-8')
    elif age < 15:
        return '10-14'.encode('utf-8')
    elif age < 20:
        return '15-19'.encode('utf-8')
    elif age < 25:
        return '20-24'.encode('utf-8')
    elif age < 30:
        return '25-29'.encode('utf-8')
    elif age < 35:
        return '30-34'.encode('utf-8')
    elif age < 40:
        return '35-39'.encode('utf-8')
    elif age < 45:
        return '40-44'.encode('utf-8')
    elif age < 50:
        return '45-49'.encode('utf-8')
    elif age < 55:
        return '50-54'.encode('utf-8')
    elif age < 60:
        return '55-59'.encode('utf-8')
    elif age < 65:
        return '60-64'.encode('utf-8')
    elif age < 70:
        return '65-69'.encode('utf-8')
    elif age < 75:
        return '70-74'.encode('utf-8')
    elif age < 80:
        return '75-79'.encode('utf-8')
    elif age < 85:
        return '80-84'.encode('utf-8')
    elif age < 90:
        return '85-89'.encode('utf-8')
    elif age < 95:
        return '90-94'.encode('utf-8')
    elif age < 100:
        return '95-99'.encode('utf-8')
    elif age >= 100:
        return '100+'.encode('utf-8')
    else:
        print('Insert age between 1-85+.')
        return


def race_encode(race):
    # Insert full name string, return utf-8 object of race code.
    race_key = {'White': '2106-3'.encode('utf-8'),
                'Asian or Pacific Islander': 'A-PI'.encode('utf-8'),
                'Black or African American': '2054-5'.encode('utf-8'),
                'American Indian or Alaska Native': '1002-5'.encode('utf-8')}

    if race not in race_key.keys():
        raise KeyError("%s not present" %race)
    else:
        return race_key[race]


def state_encode(state):
    state_dict = {'Alabama': 1,
                  'Alaska': 2,
                  'Arizona': 4,
                  'Arkansas': 5,
                  'California': 6,
                  'Colorado': 8,
                  'Connecticut': 9,
                  'Delaware': 10,
                  'District of Columbia': 11,
                  'Florida': 12,
                  'Georgia': 13,
                  'Hawaii': 15,
                  'Idaho': 16,
                  'Illinois': 17,
                  'Indiana': 18,
                  'Iowa': 19,
                  'Kansas': 20,
                  'Kentucky': 21,
                  'Louisiana': 22,
                  'Maine': 23,
                  'Maryland': 24,
                  'Massachusetts': 25,
                  'Michigan': 26,
                  'Minnesota': 27,
                  'Mississippi': 28,
                  'Missouri': 29,
                  'Montana': 30,
                  'Nebraska': 31,
                  'Nevada': 32,
                  'New Hampshire': 33,
                  'New Jersey': 34,
                  'New Mexico': 35,
                  'New York': 36,
                  'North Carolina': 37,
                  'North Dakota': 38,
                  'Ohio': 39,
                  'Oklahoma': 40,
                  'Oregon': 41,
                  'Pennsylvania': 42,
                  'Rhode Island': 44,
                  'South Carolina': 45,
                  'South Dakota': 46,
                  'Tennessee': 47,
                  'Texas': 48,
                  'Utah': 49,
                  'Vermont': 50,
                  'Virginia': 51,
                  'Washington': 53,
                  'West Virginia': 54,
                  'Wisconsin': 55,
                  'Wyoming': 56}
    if state not in state_dict.keys():
        raise KeyError('%s not in states' % state)
    else:
        return state_dict[state]


def hispanic_encode(hispanic):
    hispanic_key = {'Not Hispanic': '2186-2'.encode('utf-8'),
                    'Hispanic': '2135-2'.encode('utf-8'),
                    'Unspecific': 'NS'.encode('utf-8')}
    if hispanic not in hispanic_key.keys():
        raise KeyError("%s not present" % hispanic)
    else:
        return hispanic_key[hispanic]


def self_core_dict(age, race, gender, hispanic, state):
    # Produces a dictionary of the person's stats for numpy manipulation.
    tester = {}
    tester.update({'age': age_encode(age)})
    tester.update({'race': race_encode(race)})
    tester.update({'gender': gender.encode('utf-8')})
    tester.update({'hispanic': hispanic_encode(hispanic)})
    tester.update({'state': str(state_encode(state)).encode('utf-8')})
    return tester


# Functions age_range_encode, mortality_core_raw used to create the total mortality matrix for the core.
def age_range_encode(age):
    #ages = ['<1', '1-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
    #        '55-59', '60-64', '65-69', '70-74', '75-79', '80-84']
    ages = ['1', '1-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
            '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95-99', '100+']

    byte_ages = [x.encode('utf-8') for x in ages]
    return byte_ages[byte_ages.index(age):]


def mortality_core_raw(person_dict, age_range):
    # Imports CDC mortality and 85~100+ population data.
    mortality_path = 'C:\\Users\Amy\Desktop\Research\data\\070617_113_causeofdeath_cancer.txt'
    mortality_data = np.genfromtxt(mortality_path,
                                   dtype=(object, object, object, object, object, object, object, '<i8', '<i8'),
                                   delimiter='\t',
                                   names=True)

    pop_85_path = 'C:\\Users\Amy\Desktop\Research\data\85to100_estimates_final.txt'
    pop_85_data = np.genfromtxt(pop_85_path,
                                dtype=(object, object, object, object, object, '<i8'),
                                delimiter='\t',
                                names=True)
    pop_85_ages = ['85-89'.encode('utf-8'), '90-94'.encode('utf-8'), '95-99'.encode('utf-8'), '100+'.encode('utf-8')]

    total_deaths_path = 'C:\\Users\Amy\Desktop\Research\data\\total_deaths.txt'
    totald_data = np.genfromtxt(total_deaths_path,
                                dtype=(object, object, object, object, object, '<i8', '<i8'),
                                delimiter='\t',
                                names=True)

    age_dict = {'85-89'.encode('utf-8'): 'A',
                '90-94'.encode('utf-8'): 'B',
                '95-99'.encode('utf-8'): 'C',
                '100+'.encode('utf-8'): 'D'}
    race_dict = {'2106-3'.encode('utf-8'): '1',
                 '1002-5'.encode('utf-8'): '2',
                 '2054-5'.encode('utf-8'): '3',
                 'A-PI'.encode('utf-8'): '4'}
    ethnicity_dict = {'2186-2'.encode('utf-8'): '0',
                      '2135-2'.encode('utf-8'): '1'}

    population_dict = dict()
    for entry in pop_85_data:
        age = entry[0]
        state = entry[1]
        gender = entry[2]
        race = entry[3]
        eth = entry[4]
        population = entry[5]

        label = age_dict[age] + state.decode('utf-8') + gender.decode('utf-8') + race_dict[race] + ethnicity_dict[eth]
        population_dict.update({label: population})

    for entry in mortality_data:
        age = entry[0]
        ethnicity = entry[2]
        if age in pop_85_ages and ethnicity != 'NS'.encode('utf-8'):
            race = entry[1]
            ethnicity = entry[2]
            state = entry[3]
            gender = entry[4]

            label = age_dict[age] + state.decode('utf-8') + gender.decode('utf-8') + race_dict[race] + ethnicity_dict[
                ethnicity]
            entry[8] = population_dict[label]

    # Produces the set of the person for comparison to mortality entries.
    person_set = {person_dict['race'], person_dict['gender'], person_dict['hispanic'], person_dict['state']}

    # Produces the dictionary of all deaths associated with the core by age.
    total_deaths_all = {age: 0 for age in age_range}
    for entry in totald_data:
        age = entry[0]
        deaths = entry[5]
        population = entry[6]

        if person_set.issubset(set(entry)) and age in age_range:
            total_deaths_all.update({age: total_deaths_all[age] + deaths})

    # Produces the list of sets of all mortalities associated with the core and total count of all deaths.
    mortalities = []
    total_deaths_selected = {age: 0 for age in age_range}
    total_population_by_age = {age: 0 for age in age_range}
    for row in mortality_data:
        age = row[0]
        mortality_name = row[5]

        if person_set.issubset(set(row)) and age in age_range:
            mortality_code = row[6]
            deaths = row[7]
            population = row[8]
            rate = row[7] / row[8] * 100000
            mortalities.append((age, mortality_name, mortality_code, deaths, population, rate))
            total_deaths_selected.update({age: total_deaths_selected[age] + deaths})
            total_population_by_age.update({age: population})

    # Converts the result from list of sets to a matrix.
    mortality_matches = np.array([tuple(x) for x in mortalities], dtype='object, object, object, <i8, <i8, <i8')
    mortality_matches.reshape((len(mortality_matches), 1))

    # Obtains list of unique mortalities.
    mortality_names = set([i[1] for i in mortality_matches])
    print('There are', len(mortality_names), 'total unique mortalities.', '\n')
    if len(mortality_names) == 0:
        print('Congrats! Not enough of you are dying. Perhaps try another state.')
    return mortality_matches, mortality_names, total_deaths_selected, total_deaths_all, total_population_by_age


# Function death_ramking used to create the top 12 mortality matrix.
def death_ranking(matches, names, cutoff_num):
    scores = {name: 0 for name in names}

    # Filters through all raw mortality rows to create a death score for each mortality.
    for entry in matches:
        current_disease = entry[1]
        deaths = entry[3]
        scores.update({current_disease: scores[current_disease] + deaths})
    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

    # Returns top cutoff number mortality entries if there are >cutoff_num death scores listed.
    if len(sorted_scores) >= cutoff_num:
        # Top cutoff_num scores and mortality names obtained.
        trim_scores = sorted_scores[0:cutoff_num]
        names = [entry[0] for entry in trim_scores]

        # Finds which rows are not in the top cutoff_num mortalities and removes them. Returns the trimmed matrix.
        to_delete = [i for i in range(len(matches)) if matches[i][1] not in names]
        trimmed_matches = np.delete(matches, to_delete, axis=0)

        return trimmed_matches, names
    else:
        names = [entry[0] for entry in sorted_scores]
        return matches, names


# Functions bar_chart, stacked_histogram, scatter_plot used for visualization.
def age_bracket_avg(person_dict, ages):
    population_path = 'C:\\Users\Amy\Desktop\Research\data\\year_age_popestimate.txt'
    population_data = np.genfromtxt(population_path,
                                    dtype=('<i8', object, object, object, object, '<i8'),
                                    delimiter='\t',
                                    names=True)
    population_dict = {age: np.array([0, 0, 0, 0, 0]) for age in ages}
    person_set = {person_dict['race'], person_dict['gender'], person_dict['hispanic'], person_dict['state']}

    ages = ['1', '1-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
            '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95-99', '100+']
    byte_ages = [x.encode('utf-8') for x in ages]
    age_min = (byte_ages.index(person_dict['age'])-1) * 5

    for entry in population_data:
        current_age = entry[0]
        if person_set.issubset(entry) and current_age >= age_min:
            age = entry[0]
            age_bracket = byte_ages[age // 5 + 1]
            age_bracket_year = age % 5
            population = entry[5]
            population_dict[age_bracket][age_bracket_year] = population

    for age, counts in population_dict.items():
        tens = (byte_ages.index(age) - 1) // 2 * 10 + (byte_ages.index(age) - 1) % 2 * 5
        dists = counts/sum(counts)
        avg = np.dot(dists, [0, 1, 2, 3, 4])
        population_dict.update({age: round((tens + avg), 2)})

    return population_dict


def age_of_death(matches, names, total_deaths_all, age_avgs, just_mortalities):
    age_list = list(age_avgs.keys())
    names_path = 'C:\\Users\Amy\Desktop\Research\data\\070617_113_listofdeaths.txt'
    names_data = np.genfromtxt(names_path,
                               dtype=(object, object),
                               delimiter='\t',
                               names=True)
    names_dict = {row[0]: row[1] for row in names_data}

    if just_mortalities:
        mortality_counts = {name: {age: 0 for age in age_list} for name in names}
        mortality_results = {}

        for entry in matches:
            age = entry[0]
            name = entry[1]
            deaths = entry[3]
            mortality_counts[name].update({age: deaths})

        for name, ages in mortality_counts.items():
            counts = np.array(list(ages.values()))
            indices = list(range(len(list(ages.values()))))
            avg_index = math.ceil(np.dot(counts/sum(counts), indices))
            mortality_results.update({names_dict[name]: age_avgs[age_list[avg_index]]})
        print('Average age of death from these mortalities:')
        for key, val in mortality_results.items():
            print(key.decode('utf-8'), ' - ', val, sep='')

        return mortality_results

    else:
        counts = np.array(list(total_deaths_all.values()))
        indices = list(range(len(list(total_deaths_all.values()))))
        avg_index = math.ceil(np.dot(counts/sum(counts), indices))
        avg_age = age_avgs[age_list[avg_index]]
        print('Average age of death: ', avg_age, '\n', sep='')

        return avg_age


def stacked_bar_chart(matches, names, total_deaths_all):
    # ABOUT: Takes top 12 mortality data and creates a stacked bar chart of them.

    # Creates the dictionary of mortality to death rate per 100,000.
    percentage = {name: 0 for name in names}

    for entry in matches:
        current_mortality = entry[1]
        if current_mortality in names:
            deaths = entry[3]
            percentage.update({current_mortality: (percentage[current_mortality] + deaths)})

    names_path = 'C:\\Users\Amy\Desktop\Research\data\\070617_113_listofdeaths.txt'
    names_data = np.genfromtxt(names_path,
                               dtype=(object, object),
                               delimiter='\t',
                               names=True)
    names_dict = {row[0]: row[1] for row in names_data}

    # Sums all the death rates then divides all individual rates by the sum to obtain each percentage of deaths.
    for disease, deaths in percentage.items():
        percentage.update({disease: int(round(deaths/sum(total_deaths_all.values())*100))})

    clean_percentage = {}
    for disease, deaths in percentage.items():
        new_key = names_dict[disease].decode('utf-8')
        clean_percentage.update({new_key: deaths})

    # Creates the stacked bar chart.
    df = pd.Series(clean_percentage, name=' ').to_frame()
    df = df.sort_values(by=' ', ascending=False).T

    matplotlib.style.use('ggplot')
    my_colors = ['#8dd3c7', '#91818c', '#bebada', '#fb8072', '#80b1d3',
                 '#fdb462', '#b3de69', '#fccde5', '#2ecc71',
                 '#abf99a', '#ffed6f', "#9b59b6"]
    colors = sns.color_palette(my_colors, n_colors=df.shape[1])
    cmap1 = LinearSegmentedColormap.from_list('my_colormap', colors)

    df.plot(kind='barh', stacked=True, colormap=cmap1)
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.subplots_adjust(top=0.94, bottom=0.70, left=0.07, right=0.92)
    plt.xlim(0, 100)
    ax = plt.gca()
    # ax.set_facecolor('#ededed')
    ax.yaxis.grid(False)
    plt.title('Percentage of deaths (ignoring age) in 2015')
    plt.savefig('stacked_barplot.svg', format='svg',
                additional_artists=[lgd], bbox_inches='tight',
                dpi=1200) # add , transparent=True if you want a clear background.
    plt.show()

    return percentage


def bar_plot(total_deaths_all, total_population_by_age, age_range):
    bar_dict = {age.decode('utf-8'): 0 for age in age_range}
    for age, rate in bar_dict.items():
        age_e = age.encode('utf-8')
        if total_population_by_age[age_e] != 0:
            bar_dict.update({age: total_deaths_all[age_e] / total_population_by_age[age_e] * 100000})

    X = np.arange(len(bar_dict))
    plt.bar(X, bar_dict.values(), align='center', width=0.9)
    plt.xticks(X, bar_dict.keys(), rotation='vertical')
    plt.subplots_adjust(top=0.94, bottom=0.56, left=0.12, right=0.60)
    ax = plt.gca()
    ax.xaxis.grid(False)
    plt.title('Death Rates Across Age')
    plt.xlabel('Age')
    plt.ylabel('Deaths per 100k')
    plt.savefig('barplot.svg',
                format='svg', bbox_inches='tight',
                dpi=1200)
    plt.show()


def stacked_histogram(matches, names, age_range, show_rate, stacked100):
    # ABOUT: Creates a fill chart of the top 12 mortalities over the age range.
    # Reference: https://stackoverflow.com/questions/40960437/using-a-custom-color-palette-in-stacked-bar-chart-python

    # Creates the array of age vs. mortality, each entry contains the respective death rate.
    bar_data = np.zeros((len(age_range), len(names)))

    for entry in matches:
        current_age = entry[0]
        current_mortality = entry[1]
        deaths = entry[3]
        population = entry[4]
        if show_rate:
            bar_data[age_range.index(current_age), names.index(current_mortality)] = deaths/population*100000
        else:
            bar_data[age_range.index(current_age), names.index(current_mortality)] = deaths

    if stacked100:
        # Rescales to sum to 100% across each age.
        sum_t = bar_data.sum(axis=1)
        for row_age in range(bar_data.shape[0]):
            # Checks if there are any at all.
            if sum_t[row_age] != 0:
                bar_data[row_age, :] = bar_data[row_age, :]/sum_t[row_age]*100

    # X-axis values based on age.
    age_labels = [age.decode('utf-8') for age in age_range]

    # Stacked histogram values of mortality names.
    names_path = 'C:\\Users\Amy\Desktop\Research\data\\070617_113_listofdeaths.txt'
    names_data = np.genfromtxt(names_path,
                               dtype=(object, object),
                               delimiter='\t',
                               names=True)
    names_dict = {row[0]: row[1] for row in names_data}
    name_labels = [names_dict[name].decode('utf-8') for name in names]

    # Labels for concatenated mortality name + data matrix for the histogram.
    bar_columns = ['Age'] + name_labels

    # Creating the stacked histogram.
    matplotlib.style.use('ggplot')
    my_colors = ['#8dd3c7', '#91818c', '#bebada', '#fb8072', '#80b1d3',
                 '#fdb462', '#b3de69', '#fccde5', '#2ecc71',
                 '#abf99a', '#ffed6f', "#9b59b6"]
    colors = sns.color_palette(my_colors, n_colors=len(name_labels))
    cmap1 = LinearSegmentedColormap.from_list('my_colormap', colors)

    bar_data = np.hstack((np.array(age_labels)[np.newaxis].T, bar_data))
    df = pd.DataFrame(data=bar_data)
    df.columns = bar_columns
    df = df.set_index('Age')
    df = df.astype(float)
    fig = df.plot(kind='bar', stacked=True, colormap=cmap1, width=0.75)
    ax = plt.gca()
    ax.xaxis.grid(False)

    # In browser: Nix legend and see mortality by hovering over w/ mouse.
    handles, labels = fig.get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.09, right=0.57)
    if stacked100:
        if show_rate:
            plt.ylabel('Deaths per 100k')
        else:
            plt.ylabel('Deaths')
        plt.title('Ranked mortality distributions across age in 2015')
        plt.savefig('stacked_histogram100.svg', format='svg', dpi=1200)
    else:
        if show_rate:
            plt.ylabel('Deaths per 100k')
        else:
            plt.ylabel('Deaths')
        plt.title('Ranked mortality deaths across age in 2015')
        plt.savefig('stacked_histogram.svg', format='svg', dpi=1200)
    plt.show()


def cohort_dr(ages, total_deaths_all, total_population):
    total_dr = {age: 0 for age in ages}
    for age, rate in total_dr.items():
        if total_population[age] != 0:
            total_dr.update({age: total_deaths_all[age]/total_population[age] * 100000})

    proportion_living = {age: 0 for age in ages}
    proportion_living.update({ages[0]: 100000})
    for age in ages:
        age_index = ages.index(age)
        if age_index != 0:
            alive = (1 - 5 * total_dr[ages[age_index-1]]/100000)*proportion_living[ages[age_index-1]]
            if alive < 0:
                alive = 0
            proportion_living.update({age: alive})

    cohort = {}
    for age, value in proportion_living.items():
        cohort.update({age.decode('utf-8'): value})

    X = np.arange(len(cohort))
    plt.bar(X, cohort.values(), align='center', width=0.9)
    plt.xticks(X, cohort.keys(), rotation='vertical')
    plt.subplots_adjust(top=0.94, bottom=0.56, left=0.12, right=0.60)
    ax = plt.gca()
    ax.xaxis.grid(False)
    plt.title('Cohort Population Across Age')
    plt.xlabel('Age')
    plt.ylabel('Population')
    plt.savefig('chr_test.svg',
                format='svg', bbox_inches='tight',
                dpi=1200)
    plt.show()

    cohort_death_rates_dict = {age: 0 for age in ages}
    #for index, age in enumerate(ages):
    #    if index == 0:
    #        cohort_death_rates_dict.update({age: total_dr[age]})
    #    else:
    #        new_rate = proportion_living[age] -

    #for age, living in proportion_living.items():
    #    if age == ages[0]:
    #        cohort_death_rates_dict.update({age: total_dr[age]})
    #    else:
    #        rate =


def disease_occur_avg(matches, names, ages):
    data = np.zeros((len(names), len(ages)))

    for entry in matches:
        current_age = entry[0]
        current_death = entry[1]
        death_rate = entry[3]

        data[names.index(current_death), ages.index(current_age)] = death_rate

    for ii in range(data.shape[0]):
        sum_t = sum(data[ii, :])
        data[ii, :] = data[ii, :]/sum_t

    age_numbers = [0, 3, 7, 12, 17, 22, 30, 40, 50, 60, 70, 80, 90]
    x_axis = age_numbers[(13 - len(ages)):13]
    averages = {name: np.dot(x_axis, data[names.index(name), :]) for name in names}

    return averages


# Function master_switch controls all prior functions.
def master_switch(age, race, gender, hispanic, state):
    # 1: Convert core stats to utf-8 dictionary. Also obtains array of ages past current age.
    print('Input:', age, race, gender, hispanic, state, '\n')
    person_dict = self_core_dict(age, race, gender, hispanic, state)
    person_ages = age_range_encode(person_dict['age'])
    cutoff_num = 12

    # 2: Matrix of all mortalities for core. Mortality names too.
    raw_data, raw_names, total_deaths_selected, total_deaths_all, total_population_by_age = \
        mortality_core_raw(person_dict, person_ages)

    # 3: Matrix of top <=20 mortalities for core. Mortality names too. Need this for visuals.
    trimmed_data, trimmed_names = death_ranking(raw_data, raw_names, cutoff_num)

    # 4: Average age of death for all deaths, then for each mortality.
    age_avgs = age_bracket_avg(person_dict, person_ages)
    avg_age_all = age_of_death(trimmed_data, trimmed_names, total_deaths_all, age_avgs, just_mortalities=False)
    avg_age_mortalities = age_of_death(trimmed_data, trimmed_names, total_deaths_all, age_avgs, just_mortalities=True)

    # 4a: Bar chart. Returns the dictionary of mortality name to percent of deaths (if needed, else delete).
    percentages = stacked_bar_chart(trimmed_data, trimmed_names, total_deaths_all)

    # 4b: Scatter plot.
    bar_plot(total_deaths_all, total_population_by_age, person_ages)

    # 4c: Stacked histogram.
    stacked_histogram(trimmed_data, trimmed_names, person_ages, show_rate=True, stacked100=False)

    # 4c: Stacked 100% histogram.
    stacked_histogram(trimmed_data, trimmed_names, person_ages, show_rate=True, stacked100=True)

    # 5: Cohort death rate.
    cohort_dr(person_ages, total_deaths_all, total_population_by_age)

    # we arbitrarily choose the first 3 diseases in the common disease list for testing the action results.
    #action_diseases = mortality_names[0:3]
    #print('Action disease targets:', action_diseases)
    #action_disease(person_dict, action_diseases, age_averages, mortality_scores, mortality_names)


# MASTER TRIGGER TRIGGER
#age_in = int(input('Age (>0): '))
#gender_in = str(input('Gender: '))
#race_in = str(input('Race (White/Asian or Pacific Islander/Black or African American/American Indian or Alaska Native): '))
#hispanic_in = str(input('Hispanic? (Not Hispanic/Hispanic): '))
#state_in = str(input('State: '))

#master_switch(age=age_in, gender=gender_in, race=race_in, hispanic=hispanic_in, state=state_in)

"""
    Race: White, Asian or Pacific Islander, Black or African American, American Indian or Alaska Native.
    State: Any state with the full spelling - no acronyms. 
    Age: Any positive integer larger than 0.
    Gender: M or F.
    Ethnicity: Hispanic or Not Hispanic.
    
    Requires all four parameters. I'd like to think that people would know all of them, but
    in the event that race is a question it's not something that can be helped due to census limitations.

    Need to implement these limitations in the GUI in the future. 
"""


def start_query():
    race_in = race_get.get()

    hispanic_in = ethnicity_get.get()
    if hispanic_in == 'Not Hispanic/Latino':
        hispanic_in = 'Not Hispanic'
    else:
        hispanic_in = 'Hispanic'

    gender_in = gender_get.get()
    if gender_in == 'Male':
        gender_in = 'M'
    else:
        gender_in = 'F'

    state_in = state_get.get()
    age_in = int(age_get.get())

    master_switch(age=age_in, gender=gender_in, race=race_in, hispanic=hispanic_in, state=state_in)

root = Tk()
root.title('Prototype GUI')

menu = Menu(root)
root.config(menu=menu, bg='white', width=40)

race_get = StringVar(root)
race_get.set('White')
race_menu = OptionMenu(root, race_get, 'White',
                       'Black or African American',
                       'American Indian or Alaska Native',
                       'Asian or Pacific Islander')
race_menu.config(width=30, relief=GROOVE, bg='#a6cee3', borderwidth=0.1)
race_menu.grid(column=2, row=2)
race_menu.pack()

gender_get = StringVar(root)
gender_get.set('Male')
gen_menu = OptionMenu(root, gender_get, 'Male', 'Female')
gen_menu.config(width=30, relief=GROOVE, bg='#b2df8a', borderwidth=0.1)
gen_menu.pack()

ethnicity_get = StringVar(root)
ethnicity_get.set('Not Hispanic/Latino')
eth_menu = OptionMenu(root, ethnicity_get, 'Not Hispanic/Latino', 'Hispanic/Latino')
eth_menu.config(width=30, relief=GROOVE, bg='#fb9a99', borderwidth=0.1)
eth_menu.pack()

state_str = StringVar()
state_get = Entry(root, textvariable=state_str)
state_get.pack()
state_str.set('State')

age_str = StringVar()
age_get = Entry(root, textvariable=age_str)
age_get.pack()
age_str.set('Age')

start = Button(root, text='Start', width=10, relief=GROOVE, command=start_query)
start.pack()

status = Label(root, text='Input your demographics!', bd=1, relief=SUNKEN, anchor=W)
status.pack(side=BOTTOM, fill=X)

root.mainloop()
