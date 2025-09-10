# decentralized-epistemic-planning

## How to use?

~~~cmd
python entrance.py -h # check the full help of this program
~~~

There is a MakeFile in the root directory. If you can use makefile in your machine, then following commands should work.

```cmd
# Illustrate the possible commands
make

# An example
make coin1 # this will run the domain coin with the problem coin/problem1

# With extra arguments
make coin1 args="..." # args="..." allows you to input extra arguments
```

Or, you can directly use following commands:

```cmd
python entrance.py -d coin/domain.pddl -p coin/problem1 -ob coin.py --strategy s-jbfs.py --rules coin.py
```

The model files are in directory `./models`. Here:

- `-d [domain_file]`: Required
- `-p [problem_folder]`: Required
- `-ob [observation_function_file]`: Required, all observation function files are in `./observation_functions`
- `--strategy [strategy_file]`: Required, all strategy files are in `./policy_strategies`
- `--rules [rule]`: Required, all rule files are in `./rules`

Make sure you are using the correct observation function, strategy, rules when you are trying to run a specific model



The following are the extra commands that you can choose

```cmd
python entrance.py [required_args] \
--cooperative # the model will run in share goals mode
-tests [num_tests] # the model will simulate num_tests times
-actions [action_file] # the model will firstly run the actions defined in the file, and start simulate after that
--log-level [log_level] # set the console log level
--log-display # the log will display in the console

# remind: the following commands are not updated, I am pretty sure that they are not useful
--generate_problem # will not simulate the model but generate all possible problems based on given model

```

## Action File

Some action files are already predefined, here explain the syntax

```txt
# ./models/coin/actions/actions1.txt
a: peek a
a: return a
b: peek b
b: flip_up b c1
```

- **You don't needs to input `models/` in any action file directory. For example, the provided file can input like `coin/actions.action1.txt`** 
- The leading words (words before `:`) is the name of agent
- Remaining part is the description of an action, I think this is really easy to understand, **but**
  - The description of action name, parameters must follow the definition of action in defined domain. For example:

```txt
# flip_up action definition in coin domain
(:action flip_up
    :parameters (?self - agent ?c - coin)
    :precondition (
        (= (peeking ?self) 1)
        (= (coin ?c) 0)
    )
    :effect (
        (assign (coin ?c) 1)
    )
)
```

Since the order here is `?self -agent ?c - coin`, the description must follow this order. So `flip_up b c1` will work but `flip_up c1 b` is not!

## Remind

- To read the code, you can start from **entrance.py**, everything start from there.
- Please read **abstract.py**
- Please put all strategy file into **policy_strategy folder**. Make sure you derive **AbstractPolicyStrategy** class, which has been defined in **abstract.py** .
- Please put all observation function file into **observation_functions fold**. Make sure you derive **AbstractObservationFunction** class, which has been defined in **abstract.py**

- **Now use complete_bfs.py as the strategy, other strategy might has some problem (random will not have any problem)**



### Progress:

- [x] Domain parser
- [x] Problem parser
- [x] Model builder
  - [x] Model checker
    - [x] Regular checker: syntax, name, entities
    - [x] Goal conflict checker: Cooperative, Neutral
  - [x] Builder
- [x] Problem Generator
  - [x] initial world generator
  - [x] goal checker
  - [x] parse the problem to pddl file

- [x] Epistemic Model
- [x] Problem Solver
- [x] Intention prediction
- [x] Virtual World generator
  - [x] Rules implementation
  - [x] All functions generator
  - [x] Filter the functions based on agent's own perspective
- [x] Epistemic Logics




## A round process:

- [x] Agent observation and update their holding functions
- [x] Agents get their successors based on their holding functions
- [x] Agents choose one of the successor that will help them to reach their goal and belief goal
- [x] check the chosen action with the ontic world functions, if it pass, then do the action, otherwise it will stay, and update their belief based on the action's condition
- [x] Agent update their belief goals based on the observation of other agent's movement
  - [x] Fixing: it may generate a set of invalid goals, need to find a way to group the conflict goals and product them with other goals

- [x] End round



# Strategies:

- [x] Random: randomly choose a valid action
- [x] BFS: generate the virtual model based on agent's own perspective, and use the centralized method to simulate the model such as BFS.
  - The only thing needs to consider to check whether the expand action is valid to the world or not.
  - The justified perspective method will be implement in this model.
  - Fully expand the successors without action filter
  - Local BFS search is also in a turn based setting
- [x] CBFS: complete BFS. This is using the same method as BFS
- [x] JBFS: Justified BFS. This method will filter the actions based on agent's experience of world-action mapper. 
  - This method expect do not have any deadlock issue
- [x] S-CBFS: Sequence-Complete BFS. Basic idea is still BFS, but the local search is not turn based
  - The first movement will only be the input agent's actions, the remaining movements will be unordered
  - Able to avoid tons of redundant expansions, massively increase the calculation speed
  - Compare to CBFS, this will implicitly avoid some deadlock issues which happenned in CBFS.
- [x] S-JBFS: Sequence-Justified BFS. Basic idea is JBFS, but the local search is not turn based
  - same idea as S-CBFS.
- [x] Sample-S-JBFS: Sample-Sequence-Justified BFS. The based idea is still S_JBFS, but this method will not simulate all generated virtual world but randomly pick half of them. This will still cause deadlock in a low probability, but massively improve the performance.
