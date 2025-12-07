;Header and description

(define
    (domain grid)

    (:types
        agent survivor location
    )

    (:functions
        (agent_loc ?a - agent)
        (survivor_loc ?s - survivor)
        (movable ?a - agent)
        (sharable ?a - agent)
        (receivable ?a - agent)
        (loc_id ?l - location)
        (searched ?l - location)
        (shared ?s - survivor)
        (connected ?l1 ?l2 - location)
        (sharing ?a - agent)
    )

    (:action move
        :parameters (?self - agent ?l1 ?l2 - location)
        :precondition (
            (= (movable ?self) 1)
            (= (agent_loc ?self) (loc_id ?l1))
            (= (connected ?l1 ?l2) 1)
        )
        :effect (
            (assign (agent_loc ?self) (loc_id ?l2))
            (assign (searched ?l2) 1)
            (assign (sharing ?self) 0)
        )
    )

    (:action move_and_share
        :parameters (?self - agent ?l1 ?l2 - location)
        :precondition (
            (= (movable ?self) 1)
            (= (agent_loc ?self) (loc_id ?l1))
            (= (connected ?l1 ?l2) 1)
            (= (sharable ?self) 1)
        )
        :effect (
            (assign (agent_loc ?self) (loc_id ?l2))
            (assign (searched ?l2) 1)
            (assign (sharing ?self) 1)
        )
    )

    (:action move_and_share_survivor
        :parameters (?self - agent ?s - survivor ?l1 ?l2 - location)
        :precondition (

            (= (movable ?self) 1)
            (= (agent_loc ?self) (loc_id ?l1))
            (= (survivor_loc ?s) (loc_id ?l2))
            (= (connected ?l1 ?l2) 1)
            (= (sharable ?self) 1)
        )
        :effect (
            (assign (agent_loc ?self) (loc_id ?l2))
            (assign (searched ?l2) 1)
            (assign (shared ?s) 1)
            (assign (sharing ?self) 1)
        )
    )
)