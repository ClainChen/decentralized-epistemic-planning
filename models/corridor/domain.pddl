;Header and description

(define
    (domain corridor)

    (:types
        item agent
    )

    (:functions
        (agent_loc ?a - agent)
        (item_loc ?i - item)
        (holding ?a - agent)
        (hold_by ?i - item ?a - agent)
        (is_free ?i - item)
    )

    (:action move_right_without_item
        :parameters (?a - agent)
        :precondition (
            (= (holding ?a) 0)
            (= (agent_loc ?a) 1)
        )
        :effect (
            (increase (agent_loc ?a) 1)
        )
    )

    (:action move_left_without_item
        :parameters (?a - agent)
        :precondition (
            (= (holding ?a) 0)
            (= (agent_loc ?a) 2)
        )
        :effect (
            (decrease (agent_loc ?a) 1)
        )
    )

    (:action move_right_with_item
        :parameters (?a - agent ?i - item)
        :precondition (
            (= (agent_loc ?a) 1)
            (= (hold_by ?i ?a) 1)
            (= (is_free ?i) 0)
        )
        :effect (
            (increase (agent_loc ?a) 1)
            (increase (item_loc ?i) 1)
        )
    )

    (:action move_left_with_item
        :parameters (?a - agent ?i - item)
        :precondition (
            (= (agent_loc ?a) 2)
            (= (hold_by ?i ?a) 1)
            (= (is_free ?i) 0)
        )
        :effect (
            (decrease (agent_loc ?a) 1)
            (decrease (item_loc ?i) 1)
        )
    )

    (:action pick_up
        :parameters (?a - agent ?i - item)
        :precondition (
            (= (agent_loc ?a) (item_loc ?i))
            (= (holding ?a) 0)
            (= (is_free ?i) 1)
        )
        :effect (
            (assign (holding ?a) 1)
            (assign (hold_by ?i ?a) 1)
            (assign (is_free ?i) 0)
        )
    )

    (:action drop_item
        :parameters (?a - agent ?i - item)
        :precondition (
            (= (holding ?a) 1)
            (= (hold_by ?i ?a) 1)
        )
        :effect (
            (assign (holding ?a) 0)
            (assign (hold_by ?i ?a) 0)
            (assign (is_free ?i) 1)
        )
    )
)