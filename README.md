# decentralized-epistemic-planning

~~~bash
python entrance.py -d corridor/domain.pddl -p corridor/problem01

python entrance.py -h # check the full help of this program
~~~

Progress:

- [x] Domain parser
- [x] Problem parser
- [x] Model builder
  - [x] Model checker
  - [x] Builder

- [ ] Problem Generator
- [ ] Epistemic Model
- [ ] Problem Solver



## A round process:

- [x] Agent observation and update their holding functions
- [x] Agents get their successors based on their holding functions
- [ ] Agents choose one of the successor that will help them to reach their goal and belief goal
- [ ] check the chosen action with the ontic world functions, if it pass, then do the action, otherwise it will stay, and update their belief based on the action's condition
- [ ] Agent update their belief goals based on the observation of other agent's movement
- [ ] End round
