(define
    (domain deliver2r)

    (:types
        item agent
    )

    (:functions
        (agent_loc ?a - agent)
        (item_id ?i - item)
        (item_loc ?i - item)
        (hold ?a - agent)
        (is_free ?i - item)
    )

    (:action move_right_without_item
        :parameters (?self - agent)
        :precondition (
            (= (hold ?self) nothing)
            (= (agent_loc ?self) 1)
        )
        :effect (
            (increase (agent_loc ?self) 1)
        )
    )

    (:action move_left_without_item
        :parameters (?self - agent)
        :precondition (
            (= (hold ?self) nothing)
            (= (agent_loc ?self) 2)
        )
        :effect (
            (decrease (agent_loc ?self) 1)
        )
    )

    (:action move_right_with_item
        :parameters (?self - agent ?i - item)
        :precondition (
            (!= (item_id ?i) nothing)
            (= (agent_loc ?self) 1)
            (= (hold ?self) (item_id ?i))
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
            (!= (item_id ?i) nothing)
            (= (agent_loc ?self) 2)
            (= (hold ?self) (item_id ?i))
            (= (is_free ?i) 0)
        )
        :effect (
            (decrease (agent_loc ?self) 1)
            (decrease (item_loc ?i) 1)
        )
    )

    (:action pick
        :parameters (?self - agent ?i - item)
        :precondition (
            (!= (item_id ?i) nothing)
            (= (agent_loc ?self) (item_loc ?i))
            (= (hold ?self) nothing)
            (= (is_free ?i) 1)
        )
        :effect (
            (assign (hold ?self) (item_id ?i))
            (assign (is_free ?i) 0)
        )
    )

    (:action drop
        :parameters (?self - agent ?i - item)
        :precondition (
            (!= (item_id ?i) nothing)
            (= (hold ?self) (item_id ?i))
            (= (is_free ?i) 0)
        )
        :effect (
            (assign (hold ?self) nothing)
            (assign (is_free ?i) 1)
        )
    )
)