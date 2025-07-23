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
        :parameters (?self - agent)
        :precondition (
            (= (holding ?self) 0)
            (= (agent_loc ?self) 1)
        )
        :effect (
            (increase (agent_loc ?self) 1)
        )
    )

    (:action move_left_without_item
        :parameters (?self - agent)
        :precondition (
            (= (holding ?self) 0)
            (= (agent_loc ?self) 2)
        )
        :effect (
            (decrease (agent_loc ?self) 1)
        )
    )

    (:action move_right_with_item
        :parameters (?self - agent ?i - item)
        :precondition (
            (= (agent_loc ?self) 1)
            (= (hold_by ?i ?self) 1)
            (= (is_free ?i) 0)
        )
        :effect (
            (increase (agent_loc ?self) 1)
            (increase (item_loc ?i) 1)
        )
    )

    (:action move_left_with_item
        :parameters (?self - agent ?i - item)
        :precondition (
            (= (agent_loc ?self) 2)
            (= (hold_by ?i ?self) 1)
            (= (is_free ?i) 0)
        )
        :effect (
            (decrease (agent_loc ?self) 1)
            (decrease (item_loc ?i) 1)
        )
    )

    (:action pick_up
        :parameters (?self - agent ?i - item)
        :precondition (
            (= (agent_loc ?self) (item_loc ?i))
            (= (holding ?self) 0)
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