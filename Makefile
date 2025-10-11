.PHONY: help coin1 coin2 corridor2a1i_1 corridor2a2i_1 corridor2a2i_2 corridor4a4i_1 corridor4a2i_1 grapevine2a2s grapevine3a2s grapevine3a2s2d grapevine4a1s1d grapevine4a1s2d grapevine8a1s1d mapf1 mapf2 mapf3 mapf4 mapf5 mapf6 matrix2a1i3r_1 matrix2a1i4r_1 clean

.DEFAULT_GOAL := help

args ?=

help:
	@echo Useful commands:
	@echo   make coin1         
	@echo   make coin2         
	@echo   make corridor2a1i_1
	@echo   make corridor2a2i_1
	@echo   make corridor2a2i_2
	@echo   make corridor4a4i_1
	@echo   make corridor4a2i_1
	@echo   make grapevine1    
	@echo   make grapevine2    
	@echo   make grapevine3    
	@echo   make grapevine4
	@echo   make grapevine5
	@echo   make grapevine6
	@echo   make mapf1         
	@echo   make mapf2     
	@echo   make mapf3
	@echo   make mapf4    
	@echo   make mapf5    
	@echo   make mapf6    
	@echo   make matrix2a1i3r_1
	@echo   make matrix2a1i4r_1
	@echo   make clean         

# --without_agt_goal --without_agt_exp

coin1:
	python entrance.py \
		-d coin/domain.pddl \
		-p coin/problem1 \
		-ob coin.py \
		--strategy experiment/share.py \
		--rules coin.py \
		$(args)

coin2:
	python entrance.py \
		-d coin/domain.pddl \
		-p coin/problem2 \
		-ob coin.py \
		--strategy experiment/filtergoalexp.py \
		--rules coin.py \
		$(args)

corridor2a1i_1:
	python entrance.py \
		-d corridor/domain.pddl \
		-p corridor/2a1i_1 \
		-ob corridor.py \
		--strategy experiment/filtergoalexp.py \
		--rules corridor.py \
		$(args)

corridor2a2i_1:
	python entrance.py \
		-d corridor/domain.pddl \
		-p corridor/2a2i_1 \
		-ob corridor.py \
		--strategy experiment/filtergoalexp.py \
		--rules corridor.py \
		$(args)

corridor2a2i_2:
	python entrance.py \
		-d corridor/domain.pddl \
		-p corridor/2a2i_2 \
		-ob corridor.py \
		--strategy experiment/share.py \
		--rules corridor.py \
		$(args)

corridor4a4i_1:
	python entrance.py \
		-d corridor/domain.pddl \
		-p corridor/4a4i_1 \
		-ob corridor.py \
		--strategy experiment/share.py \
		--rules corridor.py \
		$(args)

corridor4a2i_1:
	python entrance.py \
		-d corridor/domain.pddl \
		-p corridor/4a2i_1 \
		-ob corridor.py \
		--strategy s-jbfs.py \
		--rules corridor.py \
		$(args)

grapevine2a2s:
	python entrance.py \
		-d grapevine/domain.pddl \
		-p grapevine/2a2s \
		-ob grapevine.py \
		--strategy experiment/share.py \
		--rules grapevine.py \
		$(args)

grapevine3a2s:
	python entrance.py \
		-d grapevine/domain.pddl \
		-p grapevine/3a2s \
		-ob grapevine.py \
		--strategy experiment/share.py \
		--rules grapevine.py \
		$(args)

grapevine3a2s2d:
	python entrance.py \
		-d grapevine/domain.pddl \
		-p grapevine/3a2s2d \
		-ob grapevine.py \
		--strategy experiment/filtergoal.py \
		--rules grapevine.py \
		$(args)

grapevine4a1s1d:
	python entrance.py \
		-d grapevine/domain.pddl \
		-p grapevine/4a1s1d \
		-ob grapevine.py \
		--strategy experiment/share.py \
		--rules grapevine.py \
		$(args)

grapevine4a1s2d:
	python entrance.py \
		-d grapevine/domain.pddl \
		-p grapevine/4a1s2d \
		-ob grapevine.py \
		--strategy experiment/share.py \
		--rules grapevine.py \
		$(args)

grapevine8a1s1d:
	python entrance.py \
		-d grapevine/domain.pddl \
		-p grapevine/8a1s1d \
		-ob grapevine.py \
		--strategy experiment/share.py \
		--rules grapevine.py \
		$(args)

mapf1:
	python entrance.py \
		-d mapf/domain.pddl \
		-p mapf/problem1 \
		-ob mapf.py \
		--strategy experiment/filtergoalexp.py \
		--rules mapf.py \
		$(args)

mapf2:
	python entrance.py \
		-d mapf/domain.pddl \
		-p mapf/problem2 \
		-ob mapf.py \
		--strategy s-jbfs.py \
		--rules mapf.py \
		$(args)

mapf3:
	python entrance.py \
		-d mapf/domain.pddl \
		-p mapf/problem3 \
		-ob mapf.py \
		--strategy experiment/filtergoal.py \
		--rules mapf.py \
		$(args)

mapf4:
	python entrance.py \
		-d mapf/domain.pddl \
		-p mapf/problem4 \
		-ob mapf.py \
		--strategy experiment/filtergoal.py \
		--rules mapf.py \
		$(args)

mapf5:
	python entrance.py \
		-d mapf/domain.pddl \
		-p mapf/problem5 \
		-ob mapf.py \
		--strategy experiment/filtergoalexp.py \
		--rules mapf.py \
		$(args)

mapf6:
	python entrance.py \
		-d mapf/domain.pddl \
		-p mapf/problem6 \
		-ob mapf.py \
		--strategy experiment/filtergoalexp.py \
		--rules mapf.py \
		$(args)

matrix2a1i3r_1:
	python entrance.py \
		-d matrix/domain.pddl \
		-p matrix/2a1i3r_1 \
		-ob matrix.py \
		--strategy experiment/filtergoalexp.py \
		--rules matrix.py \
		$(args)

matrix2a1i4r_1:
	python entrance.py \
		-d matrix/domain.pddl \
		-p matrix/2a1i4r_1 \
		-ob matrix.py \
		--strategy s-jbfs.py \
		--rules matrix.py \
		$(args)

clean:
	rm -f *.pyc __pycache__/*.pyc