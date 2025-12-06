;Header and description

(define
    (domain light_bulb)

    (:types
        agent light button bs
    )

    (:functions
        (light_id ?l - light)
        (bs_id ?bs - bs)
        (agent_id ?a - agent)
        (tell_lock)
        (change_lock)
        (telling ?a - agent)
        (a_observable ?l - light)
        (observable ?l - light)

        (light_state ?l - light)
        (connected ?b - button)
        (button_state ?b - button)
        (button_light_state ?b - button ?bs - bs)
    )

    (:action press_on_button
        :parameters (?self - agent ?b - button)
        :precondition (
            (= (tell_lock) 0)
            (= (change_lock) 0)
            (= (agent_id ?self) a)
            (= (button_state ?b) off)
        )
        :effect (
            (assign (button_state ?b) on)
            (assign (change_lock) 1)
        )
    )

    (:action press_off_button
        :parameters (?self - agent ?b - button)
        :precondition (
            (= (tell_lock) 0)
            (= (change_lock) 0)
            (= (agent_id ?self) a)
            (= (button_state ?b) on)
        )
        :effect (
            (assign (button_state ?b) off)
            (assign (change_lock) 1)
        )
    )

    (:action change_light_state
        :parameters (?self - agent ?b - button ?l - light ?bs - bs)
        :precondition (
            (= (tell_lock) 0)
            (= (change_lock) 1)
            (= (agent_id ?self) external)
            (= (connected ?b) (light_id ?l))
            (= (button_state ?b) (bs_id ?bs))
            (!= (light_state ?l) (button_light_state ?b ?bs))
        )
        :effect (
            (assign (light_state ?l) (button_light_state ?b ?bs))
            (assign (change_lock) 0)
        )
    )

    (:action tell
        :parameters (?self - agent ?l - light)
        :precondition (
            (!= (agent_id ?self) a)
            (!= (agent_id ?self) external)
            (= (observable ?l) (agent_id ?self))
            (= (a_observable ?l) 0)
            (= (telling ?self) 0)
            (= (tell_lock) 0)
            (= (change_lock) 0)
        )
        :effect (
            (assign (telling ?self) 1)
            (assign (tell_lock) 1)
            (assign (a_observable ?l) 1)
        )
    )

    (:action quiet
        :parameters (?self - agent ?l - light)
        :precondition (
            (!= (agent_id ?self) a)
            (!= (agent_id ?self) external)
            (= (observable ?l) (agent_id ?self))
            (= (a_observable ?l) 1)
            (= (telling ?self) 1)
            (= (tell_lock) 1)
            (= (change_lock) 0)
        )
        :effect (
            (assign (telling ?self) 0)
            (assign (tell_lock) 0)
            (assign (a_observable ?l) 0)
        )
    )

    (:action normal_stay
        :parameters (?self - agent)
        :precondition (
            (= (tell_lock) 0)
            (= (change_lock) 0)
        )
        :effect (

        )
    )

    (:action lock_stay
        :parameters (?self - agent)
        :precondition (
            (= (tell_lock) 1)
            (= (telling ?self) 0)
            (= (change_lock) 0)
        )
        :effect (

        )
    )
    
    
    
    
)