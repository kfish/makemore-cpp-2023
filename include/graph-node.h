#pragma once

#include <iomanip>
#include <utility>

#include "node.h"

namespace ai {

class Trace {
    public:
        Trace(const Node& root)
        {
            build(root);
        }

        const std::set<NodeValue*>& nodes() const {
            return nodes_;
        }

        const std::set<std::pair<NodeValue*, NodeValue*>> edges() const {
            return edges_;
        }

    private:
        void build(const Node& v) {
            if (!nodes_.contains(v.get())) {
                nodes_.insert(v.get());
                for (auto && c : v->children()) {
                    edges_.insert({c.get(), v.get()});
                    build(c);
                }
            }
        }

    private:
        std::set<NodeValue*> nodes_{};
        std::set<std::pair<NodeValue*, NodeValue*>> edges_{};
};

class NodeName {
    public:
        NodeName(const NodeValue* ptr)
            : ptr_(ptr)
        {}

        const NodeValue* get() const {
            return ptr_;
        }

    private:
        const NodeValue* ptr_;
};

static inline std::ostream& operator<<(std::ostream& os, const NodeName& node) {
    return os << "\"node" << node.get() << "\"";
}

class NodeOp {
    public:
        NodeOp(const NodeValue* ptr)
            : ptr_(ptr)
        {}

        const NodeValue* get() const {
            return ptr_;
        }

    private:
        const NodeValue* ptr_;
};

static inline std::ostream& operator<<(std::ostream& os, const NodeOp& node) {
    return os << "\"node" << node.get() << node.get()->op() << "\"";
}

class Graph {
    public:
        Graph(const std::shared_ptr<NodeValue>& root)
            : trace_(root)
        {
        }

        std::ostream& dump(std::ostream& os) const {
            auto old_precision = os.precision();

            os << "digraph G {\n"
               << "  rankdir = \"LR\";"
               << std::endl;

            os << std::fixed << std::setprecision(4);

            for (const NodeValue* node : trace_.nodes()) {
                // For any value in the graph, create a rectangular ("record") node
                // for it
                os << "  " << NodeName(node)
                   << " [label = \"{ " << node->label()
                   << " | data=" << node->data()
                   << " | grad=" << node->grad()
                   << " }\", shape=\"record\"]"
                   << std::endl;

                if (!node->op().empty()) {
                    // If this value is the result of an operation, create
                    // an op node for it
                    os << "  " << NodeOp(node)
                       << " [label = \"" << node->op() << "\"]"
                       << std::endl;

                    // And connect the op to it
                    os << "  " << NodeOp(node)
                       << " -> " << NodeName(node) << ";"
                       << std::endl;
                }
            }

            // Edges
            for (auto && [n1, n2] : trace_.edges()) {
                // Connect n1 to the op node of n2
                os << "  " << NodeName(n1) << " -> " << NodeOp(n2) << ";" << std::endl;
            }

            os << "}" << std::endl;

            os << std::setprecision(old_precision);

            return os;
        }

    private:
        Trace trace_;
};

static inline std::ostream& operator<<(std::ostream& os, const Graph& graph) {
    return graph.dump(os);
}

}
