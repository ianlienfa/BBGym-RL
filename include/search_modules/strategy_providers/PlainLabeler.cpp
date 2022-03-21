#include "search_modules/strategy_providers/PlainLabeler.h"

PlainLabeler::PlainLabeler(){}

#if LABELER == labeler_unify
int PlainLabeler::operator()(const OneRjSumCjNode &node) const {
    return 0;
}
#elif LABELER == labeler_bylevel
int PlainLabeler::operator()(const OneRjSumCjNode &node) const {
    return node.seq.size();
}
#endif
