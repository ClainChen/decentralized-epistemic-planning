;Header and description

(define
    (domain matrix)

    (:types
        item agent room
    )

    (:functions
        (agent_loc ?a - agent ?loc - room)
        (item_loc ?i - item ?loc - room)
        (connected ?loc1 ?loc2 - room)
        (holding ?a - agent)
        (hold_by ?i - item ?a - agent)
        (is_free ?i - item)
    )

    (:action move_without_item
        :parameters (?self - agent ?from ?to - room)
        :precondition (
            (= (holding ?self) 0)
            (= (agent_loc ?self ?from) 1)
            (= (connected ?from ?to) 1)
        )
        :effect (
            (assign (agent_loc ?self ?to) 1)
            (assign (agent_loc ?self ?from) 0)
        )
    )

    (:action move_with_item
        :parameters (?self - agent ?i - item ?from ?to - room)
        :precondition (
            (= (agent_loc ?self ?from) 1)
            (= (item_loc ?i ?from) 1)
            (= (holding ?self) 1)
            (= (hold_by ?i ?self) 1)
            (= (is_free ?i) 0)
            (= (connected ?from ?to) 1)
        )
        :effect (
            (assign (agent_loc ?self ?from) 0)
            (assign (agent_loc ?self ?to) 1)
            (assign (item_loc ?i ?from) 0)
            (assign (item_loc ?i ?to) 1)
        )
    )

    (:action pick_up
        :parameters (?self - agent ?i - item ?loc - room)
        :precondition (
            (= (agent_loc ?self ?loc) 1)
            (= (item_loc ?i ?loc) 1)
            (= (holding ?self) 0)
            (= (hold_by ?i ?self) 0)
            (= (is_free ?i) 1)
        )
        :effect (
            (assign (holding ?self) 1)
            (assign (hold_by ?i ?self) 1)
            (assign (is_free ?i) 0)
        )
    )

    (:action drop_item
        :parameters (?self - agent ?i - item)
        :precondition (
            (= (holding ?self) 1)
            (= (hold_by ?i ?self) 1)
            (= (is_free ?i) 0)
        )
        :effect (
            (assign (holding ?self) 0)
            (assign (hold_by ?i ?self) 0)
            (assign (is_free ?i) 1)
        )
    )
)