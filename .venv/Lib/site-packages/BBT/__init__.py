
import random



def get_codes():
    R_LIS={ 
    'rock' : 0,
    'spock': 1,
    'paper': 2,
    'lizard': 3,
    'scissors': 4}
    return R_LIS

def name_to_number(name):
    if name == 'paper':
        return 2
    elif name == 'rock':
        return 0
    elif name == 'scissors':
        return 4
    elif name == 'spock':
        return 1
    elif name == 'lizard':
        return 3
    else:
        return "not a valid choice"
    


def number_to_name(number):
    if number == 0:
        return 'rock'
    elif number == 1:
        return 'spock'
    elif number == 2:
        return 'paper'
    elif number == 3:
        return 'lizard'
    elif number == 4:
        return 'scissors'
    else:
        return " inavlid choice "
    

def rpsls(player_number): 
    player_number=player_number.lower()
    player_number = name_to_number(player_number)
    
    comp_number = random.randrange(0,5)

    comp_number = number_to_name(comp_number)

    comp_number = name_to_number(comp_number)

    x = (comp_number - player_number) % 5

    # use if/elif/else to determine winner, print winner message
    if x ==   1:
        return "You have choosen: "+str(player_number)+" AND Sheldon choosed: "+str(comp_number)+" , Result: Sheldon Won!"
        
    elif x == 2:
        return "You have choosen: "+str(player_number)+" AND Sheldon choosed: "+str(comp_number)+" , Result: Sheldon Won!"
        
    elif x == 3:
        return "You have choosen: "+str(player_number)+" AND Sheldon choosed: "+str(comp_number)+" , Result: You have Won!"
        
    elif x == 4:
        return "You have choosen: "+str(player_number)+" AND Sheldon choosed: "+str(comp_number)+" , Result: You have Won!"
    else:
        return "You have choosen: "+str(player_number)+" AND Sheldon choosed: "+str(comp_number)+" , Result: Player and Sheldon tie!"


