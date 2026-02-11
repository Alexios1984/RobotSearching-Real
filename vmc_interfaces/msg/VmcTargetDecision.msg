std_msgs/Header header
geometry_msgs/Point selected_target
bool is_risky_pool            # True se scelto dalla pool "risky" (vicino al deadlock)
bool is_deadlocked            # True se il sistema ritiene di essere in deadlock
float64 score_total
float64 score_frontality      # Contributo parziale
float64 score_distance        # Contributo parziale
float64 score_relevance       # Contributo parziale
float64 exploration_percentage # % mappa esplorata corrente